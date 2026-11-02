import os
import sys
import re
import subprocess
import tempfile
import struct
import numpy as np

NGSPICE_CMD   = "ngspice"
NGSPICE_FLAGS = ["-b", "-q"]
IVERILOG_CMD  = "iverilog"
VVP_CMD       = "vvp"

# Embedded Verilog uses `real` accumulators and exports raw IEEE-754 64-bit hex via $realtobits
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

reg [2047:0] py_in_filename;
reg [2047:0] v_out_filename;

initial begin
    max_col_seen = 0;
    for (i = 0; i < 256; i = i + 1) begin
        accumulators[i] = 0.0;
        column_active[i] = 1'b0;
    end
    
    if (!$value$plusargs("PY_IN=%s", py_in_filename)) begin
        $display("Error: Missing +PY_IN=<filename> argument.");
        $finish;
    end
    if (!$value$plusargs("V_OUT=%s", v_out_filename)) begin
        $display("Error: Missing +V_OUT=<filename> argument.");
        $finish;
    end
    
    file_in = $fopen(py_in_filename, "r");
    if (file_in == 0) begin
        $display("Error: Could not open %s", py_in_filename);
        $finish;
    end
    
    // Safely read floating point inputs (%f) to avoid 32-bit integer truncation
    scan_status = $fscanf(file_in, "%d %f %d\\n", col_idx, current_in, shift_amount);
    while (scan_status == 3) begin
        accumulators[col_idx] = accumulators[col_idx] + (current_in * (64'd1 << shift_amount));
        column_active[col_idx] = 1'b1;
        if (col_idx > max_col_seen) max_col_seen = col_idx;
        scan_status = $fscanf(file_in, "%d %f %d\\n", col_idx, current_in, shift_amount);
    end
    $fclose(file_in);

    file_out = $fopen(v_out_filename, "w");
    for (i = 0; i <= max_col_seen; i = i + 1) begin
        if (column_active[i] == 1'b1) begin
            // CHANGE 1: Export raw IEEE-754 64-bit float representation in Hex format (%x) via $realtobits
            $fdisplay(file_out, "\%d \%x", i, $realtobits(accumulators[i]));
        end
    end
    $fclose(file_out);$finish;
end
endmodule
"""

def build_netlist_with_alter(W_slice: np.ndarray, x_2c: np.ndarray, rows: int, cols: int,
                             LRS: float, HRS: float, V_READ: float, slices: int) -> str:
    lines = []
    lines.append("* Resistor Crossbar MVM - NgSPICE .alter Optimization (Ideal Wires)")
    lines.append(f"* Rows={rows}, Cols={cols}\n")

    x_slice_0 = extract_bit_slice(x_2c, 0)
    x_volts_0 = x_slice_0.astype(float) * V_READ
    
    lines.append("* -- Input voltage sources --")
    for i in range(rows):
        lines.append(f"V_in_{i+1} row_{i+1} 0 DC {float(x_volts_0[i]):.6g}")
    lines.append("")

    lines.append("* -- Crossbar Resistors (Ideal Connections) --")
    for i in range(rows):
        for j in range(cols):
            r_val = HRS if W_slice[i][j] == 0 else LRS
            lines.append(f"R_r{i+1}c{j+1} row_{i+1} col_{j+1} {r_val:.6g}")
    lines.append("")

    lines.append("* -- Ammeters --")
    for j in range(cols):
        lines.append(f"V_ammeter_{j+1} col_{j+1} 0 DC 0")
    lines.append("")

    ammeter_list = " ".join(f"I(V_ammeter_{j+1})" for j in range(cols))
    
    lines.append("* -- Control Block (Looping through Input Voltages) --")
    lines.append(".control")
    
    for x_bit in range(slices):
        lines.append(f"echo 'X_BIT_MARKER: {x_bit}'")
        
        x_slice = extract_bit_slice(x_2c, x_bit)
        x_volts = x_slice.astype(float) * V_READ
        
        for i in range(rows):
            lines.append(f"alter V_in_{i+1} dc = {float(x_volts[i]):.6g}")
            
        lines.append("op")
        lines.append(f"print {ammeter_list}")
        
    lines.append(".endc")
    lines.append("\n.end")
    return "\n".join(lines)

def run_ngspice(netlist_str: str, temp_dir: str) -> str:
    with tempfile.NamedTemporaryFile(dir=temp_dir, mode="w", suffix=".cir", delete=False) as f:
        f.write(netlist_str)
        netlist_path = f.name
    log_path = netlist_path + ".log"
    try:
        cmd = [NGSPICE_CMD] + NGSPICE_FLAGS + ["-o", log_path, netlist_path]
        subprocess.run(cmd, capture_output=True, text=True, timeout=None)
        if os.path.exists(log_path):
            with open(log_path, "r") as f_log:
                return f_log.read()
        raise RuntimeError("NgSPICE log missing.")
    finally:
        if os.path.exists(netlist_path): os.unlink(netlist_path)
        if os.path.exists(log_path): os.unlink(log_path)

def parse_currents_batch(ngspice_output: str, cols: int) -> dict:
    currents_by_xbit = {}
    blocks = ngspice_output.split('X_BIT_MARKER: ')
    
    for block in blocks[1:]: 
        lines = block.strip().split('\n')
        x_bit = int(lines[0].strip()) 
        
        currents = np.zeros(cols)
        clean_out = block.replace("Using SPARSE 1.3 as Direct Linear Solver", "")
        clean_out = re.sub(r'\s+', '', clean_out)
        
        for j in range(cols):
            pattern = r"i\(v_ammeter_{}\)=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)".format(j+1)
            match = re.search(pattern, clean_out, re.IGNORECASE)
            if match:
                currents[j] = float(match.group(1))
                
        currents_by_xbit[x_bit] = currents
        
    return currents_by_xbit

def extract_bit_slice(matrix: np.ndarray, bit_pos: int) -> np.ndarray:
    return ((matrix >> bit_pos) & 1).astype(int)

# CHANGE 2: Floating-point ADC quantization maintaining physical step resolution
def adc_quantise(val: float, step: float) -> float:
    # 1. Inject a physical +0.5 LSB analog voltage bias
    analog_offset = step / 2.0
    # 2. The comparator strictly truncates (floors) the biased signal
    clean_ratio = (val + analog_offset) / step
    return float(np.floor(clean_ratio) * step)

def to_2s_complement(matrix: np.ndarray, bits: int) -> np.ndarray:
    return np.where(matrix < 0, (1 << bits) + matrix, matrix)

def execute_hardware_layer(W_layer: np.ndarray, x_input: np.ndarray, 
                           rows: int, cols: int, slices: int, 
                           LRS: float, HRS: float, V_READ: float, 
                           layer_name: str, temp_dir: str, img_idx: int) -> np.ndarray:
    
    total_slices = slices * slices
    print(f"\n[IMG {img_idx}] [{layer_name}] Simulating Analog Crossbar ({rows}x{cols})...")

    W_2c = to_2s_complement(W_layer, slices)
    x_2c = to_2s_complement(x_input, slices)

    # CHANGE 3: Separate ADC_STEP and SF definitions
    SCALING_FACTOR = (1 / V_READ)
    ADC_STEP = 1 / LRS 
    SF = LRS            # Scale factor applied ONLY on Verilog return to recover digital logit range

    slice_counter = 0
    cycle_slice_data = []

    for w_bit in range(slices):      
        W_slice = extract_bit_slice(W_2c, w_bit)
        nl = build_netlist_with_alter(W_slice, x_2c, rows, cols, LRS, HRS, V_READ, slices)
        log_data = run_ngspice(nl, temp_dir)
        batch_results = parse_currents_batch(log_data, cols)
        
        for x_bit in range(slices):
            slice_counter += 1
            combined_shift = x_bit + w_bit

            is_w_msb = (w_bit == slices - 1)
            slice_sign = -1.0 if is_w_msb else 1.0  
            
            raw_currents = batch_results[x_bit]
            cycle_slice_data.append((combined_shift, raw_currents, slice_sign))

    print(f"[IMG {img_idx}] [{layer_name}] Analog SPICE complete. Running Verilog ALU...")

    # CHANGE 4: Send float currents (with ADC step quantization) to Verilog
    py_to_v_path = os.path.join(temp_dir, "python_to_verilog.txt")
    with open(py_to_v_path, "w") as f_out:
        for combined_shift, raw_currents, slice_sign in cycle_slice_data:
            curr_scaled = raw_currents * SCALING_FACTOR
            for col_idx in range(cols):
                quantized_mag = adc_quantise(curr_scaled[col_idx], ADC_STEP)
                val_to_write = quantized_mag * slice_sign
                f_out.write(f"{col_idx} {val_to_write} {combined_shift}\n")

    v_tb_path = os.path.join(temp_dir, "shift_add_tb.v")
    with open(v_tb_path, "w") as f_tb:
        f_tb.write(VERILOG_TB_CODE)

    subprocess.run(f"{IVERILOG_CMD} -g2012 -o sim.vvp shift_add_tb.v", shell=True, check=True, cwd=temp_dir, stdout=subprocess.DEVNULL)
    subprocess.run(f"{VVP_CMD} sim.vvp +PY_IN=python_to_verilog.txt +V_OUT=verilog_to_python.txt", shell=True, check=True, cwd=temp_dir, stdout=subprocess.DEVNULL)

    hardware_accumulated = np.zeros(cols)
    v_to_py_path = os.path.join(temp_dir, "verilog_to_python.txt")
    
    if os.path.exists(v_to_py_path):
        with open(v_to_py_path, "r") as f_in:
            for line in f_in:
                if line.strip():
                    parts = line.split()
                    col_idx = int(parts[0])
                    hex_val = parts[1]
                    
                    # CHANGE 5: Unpack lossless 64-bit IEEE 754 float hex from Verilog
                    bits = int(hex_val, 16)
                    verilog_val = struct.unpack('>d', struct.pack('>Q', bits))[0]
                    
                    # CHANGE 6: Apply Scale Factor (SF) after receiving lossless float from Verilog
                    hardware_accumulated[col_idx] = verilog_val * SF
    else:
        raise RuntimeError(f"Verilog output file not found in {temp_dir}!")
    
    return hardware_accumulated

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 isolated_parallel_inference_pipeline_2.py <image_index> <bit_width> <HRS_value>")
        sys.exit(1)
        
    IMG_IDX = int(sys.argv[1])
    SLICES = int(sys.argv[2])
    HRS = float(sys.argv[3])
    
    DATA_DIR = f"mnist_mlp_data_{SLICES}"
    
    INPUT_SIZE = 784
    HIDDEN_SIZE = 64
    OUTPUT_SIZE = 10
    
    LRS            = 5.0e3
    V_READ         = 0.1
    
    img_filename = ""
    float_img_filename = f"{DATA_DIR}/sample_img_{IMG_IDX}_float.txt"
    true_label = -1
    
    if not os.path.exists(DATA_DIR):
        print(f"[IMG {IMG_IDX}] ERROR: Directory {DATA_DIR} not found.")
        sys.exit(1)
        
    for f in os.listdir(DATA_DIR):
        match = re.search(f"sample_img_{IMG_IDX}_label_(\d+)\\.txt", f)
        if match:
            img_filename = os.path.join(DATA_DIR, f)
            true_label = int(match.group(1))
            break

    try:
        W1_full = np.loadtxt(f"{DATA_DIR}/W1_matrix.txt", dtype=int)
        B1_full = np.loadtxt(f"{DATA_DIR}/B1_vector.txt", dtype=int)
        W2_full = np.loadtxt(f"{DATA_DIR}/W2_matrix.txt", dtype=int)
        B2_full = np.loadtxt(f"{DATA_DIR}/B2_vector.txt", dtype=int)
        
        W1_float = np.loadtxt(f"{DATA_DIR}/W1_float.txt", dtype=float)
        B1_float = np.loadtxt(f"{DATA_DIR}/B1_float.txt", dtype=float)
        W2_float = np.loadtxt(f"{DATA_DIR}/W2_float.txt", dtype=float)
        B2_float = np.loadtxt(f"{DATA_DIR}/B2_float.txt", dtype=float)
        
        x_full = np.loadtxt(img_filename, dtype=int)
        x_float = np.loadtxt(float_img_filename, dtype=float)
    except FileNotFoundError:
        print(f"[IMG {IMG_IDX}] ERROR: Could not find MLP data files.")
        sys.exit(1)

    nn_float_hidden = np.maximum(0, (x_float @ W1_float) + B1_float)
    nn_float_output = (nn_float_hidden @ W2_float) + B2_float
    nn_model_prediction = np.argmax(nn_float_output)

    ideal_hidden = (W1_full.T @ x_full) + B1_full
    ideal_hidden_relu = np.maximum(0, ideal_hidden)

    ram_disk_dir = "/dev/shm" if os.path.exists("/dev/shm") else None
    with tempfile.TemporaryDirectory(prefix=f"mnist_sim_{IMG_IDX}_", dir=ram_disk_dir) as temp_dir:
        
        hw_hidden = execute_hardware_layer(W1_full, x_full, INPUT_SIZE, HIDDEN_SIZE, SLICES, 
                                           LRS, HRS, V_READ, "LAYER 1", temp_dir, IMG_IDX)
        hw_hidden += B1_full

        hw_hidden_relu = np.maximum(0, hw_hidden)
        max_act = np.max(hw_hidden_relu)
        
        MAX_PIXEL_VAL = min((1 << SLICES) - 1, 255)
        
        if max_act > 0:
            hw_hidden_relu_int = np.round((hw_hidden_relu / max_act) * MAX_PIXEL_VAL).astype(int)
            ideal_hidden_relu_scaled = np.round((ideal_hidden_relu.astype(float) / float(max_act)) * MAX_PIXEL_VAL).astype(int)
        else:
            hw_hidden_relu_int = np.zeros_like(hw_hidden_relu, dtype=int)
            ideal_hidden_relu_scaled = np.zeros_like(ideal_hidden_relu, dtype=int)

        ideal_output_scaled = (W2_full.T @ ideal_hidden_relu_scaled) + B2_full

        hw_output = execute_hardware_layer(W2_full, hw_hidden_relu_int, HIDDEN_SIZE, OUTPUT_SIZE, SLICES, 
                                           LRS, HRS, V_READ, "LAYER 2", temp_dir, IMG_IDX)
        hw_output += B2_full

    hw_prediction = np.argmax(hw_output)
    ideal_prediction = np.argmax(ideal_output_scaled)

    print("\n" + "=" * 90)
    print(f"=== [IMAGE {IMG_IDX}] MNIST MLP RESULTS ({SLICES}-BIT) ===")
    print("=" * 90)
    print(f"  {'Digit':<8} | {'Hardware Verilog':>20} | {'Ideal Python Score':>20} | {'Float NN Score':>16} | {'Prediction Tag'}")
    print("  " + "-" * 86)

    for col_idx in range(OUTPUT_SIZE):
        verilog_val = hw_output[col_idx]
        ideal_val = int(ideal_output_scaled[col_idx])
        float_val = float(nn_float_output[col_idx])
        
        tags = []
        if col_idx == hw_prediction: tags.append("HW")
        if col_idx == ideal_prediction: tags.append("IDEAL")
        if col_idx == nn_model_prediction: tags.append("FLOAT")
            
        tag_str = f"<-- {', '.join(tags)} PRED" if tags else ""
        print(f"  Digit {col_idx:<2} | {verilog_val:>20.2f} | {ideal_val:>20d} | {float_val:>16.4f} | {tag_str}")
        
    print("=" * 90)
    print(f"  TRUE LABEL                  : {true_label}")
    print(f"  FLOAT NN MODEL PREDICTION   : {nn_model_prediction}")
    print(f"  IDEAL Python PREDICTION     : {ideal_prediction}")
    print(f"  HARDWARE VERILOG PREDICTION : {hw_prediction}")
    print("=" * 90)
