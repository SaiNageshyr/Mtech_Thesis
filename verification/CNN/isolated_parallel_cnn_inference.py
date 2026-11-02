import os
import sys
import re
import subprocess
import tempfile
import struct
import numpy as np
import shutil

NGSPICE_CMD   = "ngspice"
NGSPICE_FLAGS = ["-b", "-q"]
IVERILOG_CMD  = "iverilog"
VVP_CMD       = "vvp"

VERILOG_TB_CODE = """
module shift_add_tb;
integer file_in;
integer file_out;
integer scan_status; 

integer col_idx;
integer shift_amount;
integer max_col_seen;
integer i;

real accumulators [0:255];
reg column_active [0:255]; 
real current_in;

initial begin
    max_col_seen = 0;
    for (i = 0; i < 256; i = i + 1) begin
        accumulators[i] = 0.0;
        column_active[i] = 1'b0;
    end
    
    file_in = $fopen("python_to_verilog.txt", "r");
    if (file_in == 0) begin
        $display("Error: Could not open python_to_verilog.txt.");
        $finish;
    end
    
    scan_status = $fscanf(file_in, "%d %f %d", col_idx, current_in, shift_amount);
    while (scan_status == 3) begin
        accumulators[col_idx] = accumulators[col_idx] + (current_in * (64'd1 << shift_amount));
        column_active[col_idx] = 1'b1;
        if (col_idx > max_col_seen) max_col_seen = col_idx;
        scan_status = $fscanf(file_in, "%d %f %d", col_idx, current_in, shift_amount);
    end
    $fclose(file_in);

    file_out = $fopen("verilog_to_python.txt", "w");
    for (i = 0; i <= max_col_seen; i = i + 1) begin
        if (column_active[i] == 1'b1) begin
            $fdisplay(file_out, "%d %x", i, $realtobits(accumulators[i]));
        end
    end
    $fclose(file_out);
    $finish;
end
endmodule
"""

def build_netlist_with_alter(W_slice: np.ndarray, x_2c: np.ndarray, rows: int, cols: int, LRS: float, HRS: float, V_READ: float, slices: int) -> str:
    lines = []
    lines.append("* Resistor Crossbar CNN MVM - NgSPICE .alter")
    lines.append(f"* Rows={rows}, Cols={cols}\n")
    x_slice_0 = extract_bit_slice(x_2c, 0)
    x_volts_0 = x_slice_0.astype(float) * V_READ
    
    lines.append("* -- Input voltage sources --")
    for i in range(rows): lines.append(f"V_in_{i+1} row_{i+1} 0 DC {float(x_volts_0[i]):.6g}")
    lines.append("\n* -- Crossbar Resistors --")
    for i in range(rows):
        for j in range(cols):
            r_val = HRS if W_slice[i][j] == 0 else LRS
            lines.append(f"R_r{i+1}c{j+1} row_{i+1} col_{j+1} {r_val:.6g}")
    lines.append("\n* -- Ammeters --")
    for j in range(cols): lines.append(f"V_ammeter_{j+1} col_{j+1} 0 DC 0")
    
    lines.append("\n.control")
    for x_bit in range(slices):
        lines.append(f"echo 'X_BIT_MARKER: {x_bit}'")
        x_volts = extract_bit_slice(x_2c, x_bit).astype(float) * V_READ
        for i in range(rows): lines.append(f"alter V_in_{i+1} dc = {float(x_volts[i]):.6g}")
        lines.append("op\nprint " + " ".join(f"I(V_ammeter_{j+1})" for j in range(cols)))
    lines.append(".endc\n.end")
    return "\n".join(lines)

def run_ngspice(netlist_str: str, temp_dir: str) -> str:
    with tempfile.NamedTemporaryFile(dir=temp_dir, mode="w", suffix=".cir", delete=False) as f:
        f.write(netlist_str)
        netlist_path = f.name
    log_path = netlist_path + ".log"
    try:
        cmd = [NGSPICE_CMD] + NGSPICE_FLAGS + ["-o", log_path, netlist_path]
        subprocess.run(cmd, capture_output=True, text=True, timeout=None)
        with open(log_path, "r") as f_log: return f_log.read()
    finally:
        if os.path.exists(netlist_path): os.unlink(netlist_path)
        if os.path.exists(log_path): os.unlink(log_path)

def parse_currents_batch(ngspice_output: str, cols: int) -> dict:
    currents_by_xbit = {}
    for block in ngspice_output.split('X_BIT_MARKER: ')[1:]:
        lines = block.strip().split('\n')
        x_bit = int(lines[0].strip())
        currents = np.zeros(cols)
        clean_out = re.sub(r'\s+', '', block.replace("Using SPARSE 1.3 as Direct Linear Solver", ""))
        for j in range(cols):
            pattern = r"(?:i\(v_ammeter_{0}\)|v_ammeter_{0}#branch)=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)".format(j+1)
            match = re.search(pattern, clean_out, re.IGNORECASE)
            if match: currents[j] = float(match.group(1))
        currents_by_xbit[x_bit] = currents
    return currents_by_xbit

def extract_bit_slice(matrix: np.ndarray, bit_pos: int) -> np.ndarray: return ((matrix >> bit_pos) & 1).astype(int)

def adc_quantise(val: float, step: float) -> float:
    analog_offset = step / 2.0
    clean_ratio = (val + analog_offset) / step
    return float(np.floor(clean_ratio) * step)

def to_2s_complement(matrix: np.ndarray, bits: int) -> np.ndarray: return np.where(matrix < 0, (1 << bits) + matrix, matrix)

def extract_patches(image_tensor: np.ndarray, kernel_size: int, stride: int):
    c, h, w = image_tensor.shape
    out_h, out_w = (h - kernel_size) // stride + 1, (w - kernel_size) // stride + 1
    patches = [image_tensor[:, i*stride:i*stride+kernel_size, j*stride:j*stride+kernel_size].flatten() for i in range(out_h) for j in range(out_w)]
    return np.array(patches), out_h, out_w

def python_conv2d_im2col(x, w_flat, b, k, stride, is_integer=False):
    patches, out_h, out_w = extract_patches(x, k, stride)
    out = np.dot(patches.astype(object), w_flat.astype(object)) + b.astype(object) if is_integer else (patches @ w_flat) + b
    return out.reshape(out_h, out_w, w_flat.shape[1]).transpose(2, 0, 1)

def execute_hardware_mvm(W_layer, x_input, rows, cols, slices, LRS, HRS, V_READ, temp_dir):
    W_2c = to_2s_complement(W_layer, slices)
    x_2c = to_2s_complement(x_input, slices)
    SCALING_FACTOR, SF, ADC_STEP = (1 / V_READ), LRS, 1 / LRS 
    cycle_slice_data = []

    for w_bit in range(slices):      
        nl = build_netlist_with_alter(extract_bit_slice(W_2c, w_bit), x_2c, rows, cols, LRS, HRS, V_READ, slices)
        batch_results = parse_currents_batch(run_ngspice(nl, temp_dir), cols)
        for x_bit in range(slices):
            slice_sign = -1.0 if ((x_bit == slices - 1) != (w_bit == slices - 1)) else 1.0  
            cycle_slice_data.append((x_bit + w_bit, batch_results[x_bit], slice_sign))

    with open(os.path.join(temp_dir, "python_to_verilog.txt"), "w") as f_out:
        for combined_shift, raw_currents, slice_sign in cycle_slice_data:
            curr_scaled = raw_currents * SCALING_FACTOR
            for col_idx in range(cols):
                val_to_write = adc_quantise(curr_scaled[col_idx], ADC_STEP) * slice_sign
                f_out.write(f"{col_idx} {val_to_write:.10f} {combined_shift}\n")

    with open(os.path.join(temp_dir, "shift_add_tb.v"), "w") as f_tb: f_tb.write(VERILOG_TB_CODE)
    subprocess.run(f"{IVERILOG_CMD} -g2012 -o sim.vvp shift_add_tb.v", shell=True, check=True, cwd=temp_dir, stdout=subprocess.DEVNULL)
    subprocess.run(f"{VVP_CMD} sim.vvp", shell=True, check=True, cwd=temp_dir, stdout=subprocess.DEVNULL)

    hardware_accumulated = np.zeros(cols)
    with open(os.path.join(temp_dir, "verilog_to_python.txt"), "r") as f_in:
        for line in f_in:
            if line.strip():
                parts = line.split()
                col_idx = int(parts[0])
                hex_val = parts[1]
                val_realbits = struct.unpack('>d', bytes.fromhex(hex_val.zfill(16)))[0]
                hardware_accumulated[col_idx] = val_realbits * SF
    return hardware_accumulated

def execute_hardware_conv_layer(W_layer, input_maps, kernel_size, stride, slices, LRS, HRS, V_READ, layer_name, temp_dir):
    patches, out_h, out_w = extract_patches(input_maps, kernel_size, stride)
    num_patches, out_channels = patches.shape[0], W_layer.shape[1]
    out_maps_flat = np.zeros((num_patches, out_channels))
    for p in range(num_patches): out_maps_flat[p] = execute_hardware_mvm(W_layer, patches[p], patches[p].shape[0], out_channels, slices, LRS, HRS, V_READ, temp_dir)
    return out_maps_flat.T.reshape((out_channels, out_h, out_w))

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 isolated_parallel_cnn_inference.py <img_index> <bit_width> <HRS_value>")
        sys.exit(1)
        
    IMG_IDX, SLICES, HRS = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
    LRS, V_READ = 5.0e3, 0.1
    DATA_DIR = f"mnist_cnn_data_{SLICES}"

    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Directory {DATA_DIR} not found.")
        sys.exit(1)

    MAX_VAL = min((1 << (SLICES - 1)) - 1, 10000)
    QAT_SCALE = float(MAX_VAL)

    W1_full = np.loadtxt(f"{DATA_DIR}/W1_matrix.txt", dtype=np.int64)
    B1_full = np.loadtxt(f"{DATA_DIR}/B1_vector.txt", dtype=np.int64)
    W2_full = np.loadtxt(f"{DATA_DIR}/W2_matrix.txt", dtype=np.int64)
    B2_full = np.loadtxt(f"{DATA_DIR}/B2_vector.txt", dtype=np.int64)
    W3_full = np.loadtxt(f"{DATA_DIR}/W3_matrix.txt", dtype=np.int64)
    B3_full = np.loadtxt(f"{DATA_DIR}/B3_vector.txt", dtype=np.int64)
    
    W1_float, B1_float = np.loadtxt(f"{DATA_DIR}/W1_float.txt", dtype=float), np.loadtxt(f"{DATA_DIR}/B1_float.txt", dtype=float)
    W2_float, B2_float = np.loadtxt(f"{DATA_DIR}/W2_float.txt", dtype=float), np.loadtxt(f"{DATA_DIR}/B2_float.txt", dtype=float)
    W3_float, B3_float = np.loadtxt(f"{DATA_DIR}/W3_float.txt", dtype=float), np.loadtxt(f"{DATA_DIR}/B3_float.txt", dtype=float)
    
    img_filename = next(f for f in os.listdir(DATA_DIR) if re.search(f"sample_img_{IMG_IDX}_label_(\d+)\\.txt", f))
    true_label = int(re.search(r'label_(\d+)\.txt', img_filename).group(1))

    x_full = np.loadtxt(os.path.join(DATA_DIR, img_filename), dtype=np.int64).reshape((1, 28, 28))
    x_float = np.loadtxt(f"{DATA_DIR}/sample_img_{IMG_IDX}_float.txt", dtype=float).reshape((1, 28, 28))

    # =========================================================================
    # FLOAT NN MODEL PREDICTION (With Equivalent Dynamic Normalization)
    # =========================================================================
    f_conv1 = np.maximum(0, python_conv2d_im2col(x_float, W1_float, B1_float, 3, 2, False))
    f_max1 = np.max(f_conv1) if np.max(f_conv1) > 0 else 1
    f_conv1_q = np.round((f_conv1 / f_max1) * MAX_VAL) / MAX_VAL 
    
    f_conv2 = np.maximum(0, python_conv2d_im2col(f_conv1_q, W2_float, B2_float, 3, 2, False))
    f_max2 = np.max(f_conv2) if np.max(f_conv2) > 0 else 1
    f_conv2_q = np.round((f_conv2 / f_max2) * MAX_VAL) / MAX_VAL 

    nn_model_prediction = np.argmax((f_conv2_q.flatten() @ W3_float) + B3_float)

    # =========================================================================
    # IDEAL PYTHON EXECUTION
    # =========================================================================
    i_conv1_raw = python_conv2d_im2col(x_full, W1_full, np.zeros_like(B1_full), 3, 2, True)
    i_conv1_scaled = np.asarray(i_conv1_raw, dtype=np.float64) / QAT_SCALE
    for c in range(i_conv1_scaled.shape[0]): i_conv1_scaled[c, :, :] += B1_full[c]
    i_conv1_relu = np.maximum(0, i_conv1_scaled)
    
    i_max1 = np.max(i_conv1_relu) if np.max(i_conv1_relu) > 0 else 1
    i_conv1_q = np.clip(np.round((i_conv1_relu / i_max1) * MAX_VAL), 0, MAX_VAL).astype(np.int64)
    
    i_conv2_raw = python_conv2d_im2col(i_conv1_q, W2_full, np.zeros_like(B2_full), 3, 2, True)
    i_conv2_scaled = np.asarray(i_conv2_raw, dtype=np.float64) / QAT_SCALE
    for c in range(i_conv2_scaled.shape[0]): i_conv2_scaled[c, :, :] += B2_full[c]
    i_conv2_relu = np.maximum(0, i_conv2_scaled)
    
    i_max2 = np.max(i_conv2_relu) if np.max(i_conv2_relu) > 0 else 1
    i_conv2_q = np.clip(np.round((i_conv2_relu / i_max2) * MAX_VAL), 0, MAX_VAL).astype(np.int64)

    i_fc3_raw = np.dot(i_conv2_q.flatten().astype(object), W3_full.astype(object))
    i_fc3_scaled = (np.asarray(i_fc3_raw, dtype=np.float64) / QAT_SCALE) + B3_full
    ideal_prediction = np.argmax(i_fc3_scaled)

    # =========================================================================
    # HARDWARE VERILOG EXECUTION 
    # =========================================================================
    ram_disk_dir = "/dev/shm" if os.path.exists("/dev/shm") else None
    with tempfile.TemporaryDirectory(prefix=f"mnist_cnn_{IMG_IDX}_", dir=ram_disk_dir) as temp_dir:
        
        hw_conv1_raw = execute_hardware_conv_layer(W1_full, x_full, 3, 2, SLICES, LRS, HRS, V_READ, "CONV 1", temp_dir)
        hw_conv1_scaled = np.asarray(hw_conv1_raw, dtype=np.float64) / QAT_SCALE
        for c in range(hw_conv1_scaled.shape[0]): hw_conv1_scaled[c, :, :] += B1_full[c]
        hw_conv1_relu = np.maximum(0, hw_conv1_scaled)
        
        h_max1 = np.max(hw_conv1_relu) if np.max(hw_conv1_relu) > 0 else 1
        hw_conv1_int = np.clip(np.round((hw_conv1_relu / h_max1) * MAX_VAL), 0, MAX_VAL).astype(np.int64)

        hw_conv2_raw = execute_hardware_conv_layer(W2_full, hw_conv1_int, 3, 2, SLICES, LRS, HRS, V_READ, "CONV 2", temp_dir)
        hw_conv2_scaled = np.asarray(hw_conv2_raw, dtype=np.float64) / QAT_SCALE
        for c in range(hw_conv2_scaled.shape[0]): hw_conv2_scaled[c, :, :] += B2_full[c]
        hw_conv2_relu = np.maximum(0, hw_conv2_scaled)
        
        h_max2 = np.max(hw_conv2_relu) if np.max(hw_conv2_relu) > 0 else 1
        hw_conv2_int = np.clip(np.round((hw_conv2_relu / h_max2) * MAX_VAL), 0, MAX_VAL).astype(np.int64)

        hw_flattened = hw_conv2_int.flatten()
        hw_fc3_raw = execute_hardware_mvm(W3_full, hw_flattened, hw_flattened.shape[0], 10, SLICES, LRS, HRS, V_READ, temp_dir)
        hw_fc3_scaled = (np.asarray(hw_fc3_raw, dtype=np.float64) / QAT_SCALE) + B3_full
        
    hw_prediction = np.argmax(hw_fc3_scaled)

    print("\n" + "=" * 90)
    print(f"=== [IMAGE {IMG_IDX}] MNIST CNN CLASSIFICATION RESULTS ({SLICES}-BIT) ===")
    print("=" * 90)
    print(f"  TRUE LABEL                  : {true_label}")
    print(f"  FLOAT NN MODEL PREDICTION   : {nn_model_prediction}")
    print(f"  IDEAL PYTHON PREDICTION     : {ideal_prediction}")
    print(f"  HARDWARE VERILOG PREDICTION : {hw_prediction}")
    print("=" * 90)
