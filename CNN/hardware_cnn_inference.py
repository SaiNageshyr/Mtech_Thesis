#working 
'''import os
import re
import subprocess
import tempfile
import numpy as np

# -----------------------------------------------------------------------------
#  NgSPICE & Iverilog Config
# -----------------------------------------------------------------------------
NGSPICE_CMD   = "ngspice"
NGSPICE_FLAGS = ["-b", "-q"]
IVERILOG_CMD  = "iverilog"
VVP_CMD       = "vvp"

MAX_VAL_32 = 10000

# -----------------------------------------------------------------------------
#  Netlist builder (Optimized with .alter)
# -----------------------------------------------------------------------------
def build_netlist_with_alter(W_slice: np.ndarray, x_2c: np.ndarray, rows: int, cols: int,
                             LRS: float, HRS: float, V_READ: float, slices: int) -> str:
    lines = []
    lines.append("* Resistor Crossbar CNN MVM - NgSPICE .alter Optimization")
    lines.append(f"* Rows={rows}, Cols={cols}\n")

    x_slice_0 = extract_bit_slice(x_2c, 0)
    x_volts_0 = x_slice_0.astype(float) * V_READ
    
    lines.append("* -- Input voltage sources --")
    for i in range(rows):
        lines.append(f"V_in_{i+1} row_{i+1} 0 DC {float(x_volts_0[i]):.6g}")
    lines.append("")

    lines.append("* -- Crossbar Resistors --")
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
    
    lines.append(".control")
    for x_bit in range(slices):
        lines.append(f"echo 'X_BIT_MARKER: {x_bit}'")
        x_slice = extract_bit_slice(x_2c, x_bit)
        x_volts = x_slice.astype(float) * V_READ
        for i in range(rows):
            lines.append(f"alter V_in_{i+1} dc = {float(x_volts[i]):.6g}")
        lines.append("op")
        lines.append(f"print {ammeter_list}")
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

def adc_quantise(val: float, step: float) -> float:
    clean_ratio = np.round(val / step, decimals=4)
    return float(np.round(clean_ratio) * step)

def to_2s_complement(matrix: np.ndarray, bits: int) -> np.ndarray:
    return np.where(matrix < 0, (1 << bits) + matrix, matrix)

# -----------------------------------------------------------------------------
#  Python CNN Mathematical Equivalents
# -----------------------------------------------------------------------------
def extract_patches(image_tensor: np.ndarray, kernel_size: int, stride: int):
    """Slides a kernel over a 3D input map and flattens each window into a 1D array."""
    c, h, w = image_tensor.shape
    out_h = (h - kernel_size) // stride + 1
    out_w = (w - kernel_size) // stride + 1
    
    patches = []
    for i in range(out_h):
        for j in range(out_w):
            patch = image_tensor[:, i*stride:i*stride+kernel_size, j*stride:j*stride+kernel_size]
            patches.append(patch.flatten())
    return np.array(patches), out_h, out_w

def python_conv2d_im2col(x, w_flat, b, k, stride, is_integer=False):
    """Calculates ideal 3D convolution using matrix multiplication."""
    patches, out_h, out_w = extract_patches(x, k, stride)
    
    if is_integer:
        # Use Python object arrays to avoid 64-bit NumPy overflow with 32-bit QAT
        out = np.dot(patches.astype(object), w_flat.astype(object)) + b.astype(object)
    else:
        out = (patches @ w_flat) + b
        
    out = out.reshape(out_h, out_w, w_flat.shape[1])
    return out.transpose(2, 0, 1) # (out_c, out_h, out_w)

# -----------------------------------------------------------------------------
#  Hardware Layer Execution Functions
# -----------------------------------------------------------------------------
def execute_hardware_mvm(W_layer, x_input, rows, cols, slices, LRS, HRS, V_READ):
    W_2c = to_2s_complement(W_layer, slices)
    x_2c = to_2s_complement(x_input, slices)

    SCALING_FACTOR = (1 / V_READ)
    SF = LRS
    ADC_STEP = 1 / LRS 
    cycle_slice_data = []
    
    ram_disk_dir = "/dev/shm" if os.path.exists("/dev/shm") else None
    temp_dir = tempfile.mkdtemp(dir=ram_disk_dir)

    for w_bit in range(slices):      
        W_slice = extract_bit_slice(W_2c, w_bit)
        nl = build_netlist_with_alter(W_slice, x_2c, rows, cols, LRS, HRS, V_READ, slices)
        log_data = run_ngspice(nl, temp_dir)
        batch_results = parse_currents_batch(log_data, cols)
        
        for x_bit in range(slices):
            combined_shift = x_bit + w_bit
            is_x_msb = (x_bit == slices - 1)
            is_w_msb = (w_bit == slices - 1)
            slice_sign = -1.0 if (is_x_msb != is_w_msb) else 1.0  
            cycle_slice_data.append((combined_shift, batch_results[x_bit], slice_sign))

    os.rmdir(temp_dir)

    with open("python_to_verilog.txt", "w") as f_out:
        for combined_shift, raw_currents, slice_sign in cycle_slice_data:
            curr_scaled = raw_currents * SCALING_FACTOR
            for col_idx in range(cols):
                quantized_mag = adc_quantise(curr_scaled[col_idx], ADC_STEP)
                val_to_write = quantized_mag * slice_sign
                f_out.write(f"{col_idx} {val_to_write} {combined_shift}\n")

    subprocess.run(f"{IVERILOG_CMD} -g2012 -o sim.vvp shift_add_tb.v", shell=True, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(f"{VVP_CMD} sim.vvp", shell=True, check=True, stdout=subprocess.DEVNULL)

    hardware_accumulated = np.zeros(cols)
    if os.path.exists("verilog_to_python.txt"):
        with open("verilog_to_python.txt", "r") as f_in:
            for line in f_in:
                if line.strip():
                    parts = line.split()
                    col_idx = int(parts[0])
                    verilog_val = float(parts[1]) * SF
                    hardware_accumulated[col_idx] = verilog_val
    return hardware_accumulated

def execute_hardware_conv_layer(W_layer, input_maps, kernel_size, stride, slices, LRS, HRS, V_READ, layer_name):
    patches, out_h, out_w = extract_patches(input_maps, kernel_size, stride)
    num_patches = patches.shape[0]
    out_channels = W_layer.shape[1]
    
    print(f"\n[{layer_name}] Extracted {num_patches} patches. Output shape will be ({out_channels}, {out_h}, {out_w})")
    
    out_maps_flat = np.zeros((num_patches, out_channels))
    for p in range(num_patches):
        print(f"  -> NgSPICE Processing Patch {p+1}/{num_patches} for {layer_name}...", end='\r')
        rows = patches[p].shape[0]
        out_maps_flat[p] = execute_hardware_mvm(W_layer, patches[p], rows, out_channels, slices, LRS, HRS, V_READ)
    
    print(f"\n  -> {layer_name} Verilog ALU Synthesis Complete.")
    return out_maps_flat.T.reshape((out_channels, out_h, out_w))

# -----------------------------------------------------------------------------
#  Main Pipeline
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    SLICES = 32 
    HRS, LRS, V_READ = 5.0e19, 5.0e3, 0.1
    
    print("\n" + "=" * 64)
    print(f"  INITIALIZING HARDWARE 3-LAYER CNN (Precision: {SLICES}-bit QAT)")
    print("=" * 64)

    # 1. Load integer 32-bit quantized weights & biases
    W1_full = np.loadtxt("mnist_cnn_data/W1_matrix.txt", dtype=np.int64)
    B1_full = np.loadtxt("mnist_cnn_data/B1_vector.txt", dtype=np.int64)
    W2_full = np.loadtxt("mnist_cnn_data/W2_matrix.txt", dtype=np.int64)
    B2_full = np.loadtxt("mnist_cnn_data/B2_vector.txt", dtype=np.int64)
    W3_full = np.loadtxt("mnist_cnn_data/W3_matrix.txt", dtype=np.int64)
    B3_full = np.loadtxt("mnist_cnn_data/B3_vector.txt", dtype=np.int64)
    
    # 2. Load float weights & biases
    W1_float = np.loadtxt("mnist_cnn_data/W1_float.txt", dtype=float)
    B1_float = np.loadtxt("mnist_cnn_data/B1_float.txt", dtype=float)
    W2_float = np.loadtxt("mnist_cnn_data/W2_float.txt", dtype=float)
    B2_float = np.loadtxt("mnist_cnn_data/B2_float.txt", dtype=float)
    W3_float = np.loadtxt("mnist_cnn_data/W3_float.txt", dtype=float)
    B3_float = np.loadtxt("mnist_cnn_data/B3_float.txt", dtype=float)
    
    img_filename = "mnist_cnn_data/sample_img_0_label_7.txt"
    float_img_filename = "mnist_cnn_data/sample_img_0_float.txt"
    true_label = int(re.search(r'label_(\d+)\.txt', img_filename).group(1))

    x_full = np.loadtxt(img_filename, dtype=np.int64).reshape((1, 28, 28))
    x_float = np.loadtxt(float_img_filename, dtype=float).reshape((1, 28, 28))

    # =========================================================================
    # IDEAL PYTHON CALCULATION (FLOAT NN & INTEGER QUANTIZED)
    # =========================================================================
    print("Calculating Ideal Python & Float NN Scores...")
    # Float NN Math
    f_conv1 = np.maximum(0, python_conv2d_im2col(x_float, W1_float, B1_float, 3, 2, False))
    f_conv2 = np.maximum(0, python_conv2d_im2col(f_conv1, W2_float, B2_float, 3, 2, False))
    f_out = (f_conv2.flatten() @ W3_float) + B3_float
    nn_model_prediction = np.argmax(f_out)

    # Ideal 32-Bit Quantized Math
    i_conv1 = np.maximum(0, python_conv2d_im2col(x_full, W1_full, B1_full, 3, 2, True))
    i_max1 = np.max(i_conv1) if np.max(i_conv1) > 0 else 1
    # Cast to float64 before rounding
    i_conv1_q = np.round(((i_conv1 / i_max1) * MAX_VAL_32).astype(np.float64)).astype(np.int64)
    
    i_conv2 = np.maximum(0, python_conv2d_im2col(i_conv1_q, W2_full, B2_full, 3, 2, True))
    i_max2 = np.max(i_conv2) if np.max(i_conv2) > 0 else 1
    # Cast to float64 before rounding
    i_conv2_q = np.round(((i_conv2 / i_max2) * MAX_VAL_32).astype(np.float64)).astype(np.int64)

    i_out = np.dot(i_conv2_q.flatten().astype(object), W3_full.astype(object)) + B3_full.astype(object)
    ideal_prediction = np.argmax(i_out)

    # =========================================================================
    # HARDWARE SPICE CALCULATION (LAYER BY LAYER)
    # =========================================================================
    # LAYER 1
    hw_conv1 = execute_hardware_conv_layer(W1_full, x_full, 3, 2, SLICES, LRS, HRS, V_READ, "CONV 1")
    for c in range(hw_conv1.shape[0]): hw_conv1[c, :, :] += B1_full[c]
    hw_conv1_relu = np.maximum(0, hw_conv1)
    
    h_max1 = np.max(hw_conv1_relu) if np.max(hw_conv1_relu) > 0 else 1
    hw_conv1_int = np.round((hw_conv1_relu / h_max1) * MAX_VAL_32).astype(np.int64)

    # LAYER 2
    hw_conv2 = execute_hardware_conv_layer(W2_full, hw_conv1_int, 3, 2, SLICES, LRS, HRS, V_READ, "CONV 2")
    for c in range(hw_conv2.shape[0]): hw_conv2[c, :, :] += B2_full[c]
    hw_conv2_relu = np.maximum(0, hw_conv2)
    
    h_max2 = np.max(hw_conv2_relu) if np.max(hw_conv2_relu) > 0 else 1
    hw_conv2_int = np.round((hw_conv2_relu / h_max2) * MAX_VAL_32).astype(np.int64)

    # LAYER 3 (FC)
    hw_flattened = hw_conv2_int.flatten()
    print("\n[FC LAYER] Simulating Analog Crossbar...")
    hw_output = execute_hardware_mvm(W3_full, hw_flattened, hw_flattened.shape[0], 10, SLICES, LRS, HRS, V_READ)
    hw_output += B3_full

    hw_prediction = np.argmax(hw_output)

    # =========================================================================
    # FINAL COMPARISON TABLE
    # =========================================================================
    print("\n" + "=" * 90)
    print("=== MNIST 32-BIT CNN CLASSIFICATION RESULTS ===")
    print("=" * 90)
    print(f"  {'Digit':<8} | {'Hardware Verilog':>20} | {'Ideal Python Score':>19} | {'Float NN Score':>16} | {'Prediction Tag'}")
    print("  " + "-" * 86)

    for col_idx in range(10):
        verilog_val = hw_output[col_idx]
        ideal_val = int(i_out[col_idx])
        float_val = float(f_out[col_idx])
        
        tags = []
        if col_idx == hw_prediction: tags.append("HW")
        if col_idx == ideal_prediction: tags.append("IDEAL")
        if col_idx == nn_model_prediction: tags.append("FLOAT")
            
        tag_str = f"<-- {', '.join(tags)} PRED" if tags else ""
        print(f"  Digit {col_idx:<2} | {verilog_val:>20.2e} | {ideal_val:>20.2e} | {float_val:>16.4f} | {tag_str}")
        
    print("=" * 90)
    print(f"  TRUE LABEL                  : {true_label}")
    print(f"  FLOAT NN MODEL PREDICTION   : {nn_model_prediction}")
    print(f"  IDEAL PYTHON PREDICTION  : {ideal_prediction}")
    print(f"  HARDWARE VERILOG PREDICTION : {hw_prediction}")
    print("=" * 90)
    
    if hw_prediction == true_label:
        print("\n>> SUCCESS! Hardware CNN crossbar correctly identified the image! <<\n")
    else:
        print("\n>> FAILED. Hardware CNN crossbar misclassified the image. <<\n")'''
        
#scaling to get the floating point and hardware values same
import os
import re
import subprocess
import tempfile
import shutil
import numpy as np

# -----------------------------------------------------------------------------
#  NgSPICE & Iverilog Config
# -----------------------------------------------------------------------------
NGSPICE_CMD   = "ngspice"
NGSPICE_FLAGS = ["-b", "-q"]
IVERILOG_CMD  = "iverilog"
VVP_CMD       = "vvp"

QAT_SCALE = 10000.0  

# -----------------------------------------------------------------------------
#  EMBEDDED SYNTHESIZABLE VERILOG TESTBENCH (Rolled back to Floats/Reals)
# -----------------------------------------------------------------------------
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
    
    scan_status = $fscanf(file_in, "%d %f %d\\n", col_idx, current_in, shift_amount);
    while (scan_status == 3) begin
        accumulators[col_idx] = accumulators[col_idx] + (current_in * (64'd1 << shift_amount));
        column_active[col_idx] = 1'b1;
        if (col_idx > max_col_seen) max_col_seen = col_idx;
        scan_status = $fscanf(file_in, "%d %f %d\\n", col_idx, current_in, shift_amount);
    end
    $fclose(file_in);

    file_out = $fopen("verilog_to_python.txt", "w");
    for (i = 0; i <= max_col_seen; i = i + 1) begin
        if (column_active[i] == 1'b1) begin
            $fdisplay(file_out, "%d %0.0f", i, accumulators[i]);
        end
    end
    $fclose(file_out);
    $finish;
end
endmodule
"""

# -----------------------------------------------------------------------------
#  Netlist builder (Optimized with .alter)
# -----------------------------------------------------------------------------
def build_netlist_with_alter(W_slice: np.ndarray, x_2c: np.ndarray, rows: int, cols: int,
                             LRS: float, HRS: float, V_READ: float, slices: int) -> str:
    lines = []
    lines.append("* Resistor Crossbar CNN MVM - NgSPICE .alter Optimization")
    lines.append(f"* Rows={rows}, Cols={cols}\n")

    x_slice_0 = extract_bit_slice(x_2c, 0)
    x_volts_0 = x_slice_0.astype(float) * V_READ
    
    lines.append("* -- Input voltage sources --")
    for i in range(rows):
        lines.append(f"V_in_{i+1} row_{i+1} 0 DC {float(x_volts_0[i]):.6g}")
    lines.append("")

    lines.append("* -- Crossbar Resistors --")
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
    
    lines.append(".control")
    for x_bit in range(slices):
        lines.append(f"echo 'X_BIT_MARKER: {x_bit}'")
        x_slice = extract_bit_slice(x_2c, x_bit)
        x_volts = x_slice.astype(float) * V_READ
        for i in range(rows):
            lines.append(f"alter V_in_{i+1} dc = {float(x_volts[i]):.6g}")
        lines.append("op")
        lines.append(f"print {ammeter_list}")
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

# -----------------------------------------------------------------------------
#  Floor ADC Logic (Rolled back to include * step scaling)
# -----------------------------------------------------------------------------
def adc_quantise(val: float, step: float) -> float:
    analog_offset = step / 2.0
    clean_ratio = (val + analog_offset) / step
    return float(np.floor(clean_ratio) * step)

def to_2s_complement(matrix: np.ndarray, bits: int) -> np.ndarray:
    return np.where(matrix < 0, (1 << bits) + matrix, matrix)

# -----------------------------------------------------------------------------
#  Python CNN Mathematical Equivalents
# -----------------------------------------------------------------------------
def extract_patches(image_tensor: np.ndarray, kernel_size: int, stride: int):
    c, h, w = image_tensor.shape
    out_h = (h - kernel_size) // stride + 1
    out_w = (w - kernel_size) // stride + 1
    
    patches = []
    for i in range(out_h):
        for j in range(out_w):
            patch = image_tensor[:, i*stride:i*stride+kernel_size, j*stride:j*stride+kernel_size]
            patches.append(patch.flatten())
    return np.array(patches), out_h, out_w

def python_conv2d_im2col(x, w_flat, b, k, stride, is_integer=False):
    patches, out_h, out_w = extract_patches(x, k, stride)
    
    if is_integer:
        out = np.dot(patches.astype(object), w_flat.astype(object))
        out = np.round(out.astype(np.float64) / QAT_SCALE) + b
    else:
        out = (patches @ w_flat) + b
        
    out = out.reshape(out_h, out_w, w_flat.shape[1])
    return out.transpose(2, 0, 1)

# -----------------------------------------------------------------------------
#  Hardware Layer Execution Functions
# -----------------------------------------------------------------------------
def execute_hardware_mvm(W_layer, x_input, rows, cols, slices, LRS, HRS, V_READ):
    W_2c = to_2s_complement(W_layer, slices)
    x_2c = to_2s_complement(x_input, slices)

    # Rolled back to old scaling mechanics
    SCALING_FACTOR = (1 / V_READ)
    SF = LRS 
    ADC_STEP = 1 / LRS 
    cycle_slice_data = []
    
    ram_disk_dir = "/dev/shm" if os.path.exists("/dev/shm") else None
    temp_dir = tempfile.mkdtemp(dir=ram_disk_dir)

    for w_bit in range(slices):      
        W_slice = extract_bit_slice(W_2c, w_bit)
        nl = build_netlist_with_alter(W_slice, x_2c, rows, cols, LRS, HRS, V_READ, slices)
        log_data = run_ngspice(nl, temp_dir)
        batch_results = parse_currents_batch(log_data, cols)
        
        for x_bit in range(slices):
            combined_shift = x_bit + w_bit
            is_x_msb = (x_bit == slices - 1)
            is_w_msb = (w_bit == slices - 1)
            slice_sign = -1.0 if (is_x_msb != is_w_msb) else 1.0  
            cycle_slice_data.append((combined_shift, batch_results[x_bit], slice_sign))

    with open(os.path.join(temp_dir, "shift_add_tb.v"), "w") as f_tb:
        f_tb.write(VERILOG_TB_CODE)

    with open(os.path.join(temp_dir, "python_to_verilog.txt"), "w") as f_out:
        for combined_shift, raw_currents, slice_sign in cycle_slice_data:
            curr_scaled = raw_currents * SCALING_FACTOR
            for col_idx in range(cols):
                quantized_mag = adc_quantise(curr_scaled[col_idx], ADC_STEP)
                val_to_write = float(quantized_mag * slice_sign)
                f_out.write(f"{col_idx} {val_to_write} {combined_shift}\n")

    subprocess.run(f"{IVERILOG_CMD} -g2012 -o sim.vvp shift_add_tb.v", shell=True, check=True, cwd=temp_dir, stdout=subprocess.DEVNULL)
    subprocess.run(f"{VVP_CMD} sim.vvp", shell=True, check=True, cwd=temp_dir, stdout=subprocess.DEVNULL)

    hardware_accumulated = np.zeros(cols)
    if os.path.exists(os.path.join(temp_dir, "verilog_to_python.txt")):
        with open(os.path.join(temp_dir, "verilog_to_python.txt"), "r") as f_in:
            for line in f_in:
                if line.strip():
                    parts = line.split()
                    col_idx = int(parts[0])
                    verilog_val = float(parts[1]) * SF
                    hardware_accumulated[col_idx] = verilog_val
    
    # Safely remove the entire populated temp directory
    shutil.rmtree(temp_dir)
    return hardware_accumulated

def execute_hardware_conv_layer(W_layer, input_maps, kernel_size, stride, slices, LRS, HRS, V_READ, layer_name):
    patches, out_h, out_w = extract_patches(input_maps, kernel_size, stride)
    num_patches = patches.shape[0]
    out_channels = W_layer.shape[1]
    
    print(f"\n[{layer_name}] Extracted {num_patches} patches. Output shape will be ({out_channels}, {out_h}, {out_w})")
    
    out_maps_flat = np.zeros((num_patches, out_channels))
    for p in range(num_patches):
        print(f"  -> NgSPICE Processing Patch {p+1}/{num_patches} for {layer_name}...", end='\r')
        rows = patches[p].shape[0]
        out_maps_flat[p] = execute_hardware_mvm(W_layer, patches[p], rows, out_channels, slices, LRS, HRS, V_READ)
    
    print(f"\n  -> {layer_name} Verilog ALU Synthesis Complete.")
    return out_maps_flat.T.reshape((out_channels, out_h, out_w))

# -----------------------------------------------------------------------------
#  Main Pipeline
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    SLICES = 32 
    HRS, LRS, V_READ = 5.0e19, 5.0e3, 0.1
    
    print("\n" + "=" * 64)
    print(f"  INITIALIZING HARDWARE 3-LAYER CNN (Precision: {SLICES}-bit QAT)")
    print("=" * 64)

    W1_full = np.loadtxt("mnist_cnn_data/W1_matrix.txt", dtype=np.int64)
    B1_full = np.loadtxt("mnist_cnn_data/B1_vector.txt", dtype=np.int64)
    W2_full = np.loadtxt("mnist_cnn_data/W2_matrix.txt", dtype=np.int64)
    B2_full = np.loadtxt("mnist_cnn_data/B2_vector.txt", dtype=np.int64)
    W3_full = np.loadtxt("mnist_cnn_data/W3_matrix.txt", dtype=np.int64)
    B3_full = np.loadtxt("mnist_cnn_data/B3_vector.txt", dtype=np.int64)
    
    W1_float = np.loadtxt("mnist_cnn_data/W1_float.txt", dtype=float)
    B1_float = np.loadtxt("mnist_cnn_data/B1_float.txt", dtype=float)
    W2_float = np.loadtxt("mnist_cnn_data/W2_float.txt", dtype=float)
    B2_float = np.loadtxt("mnist_cnn_data/B2_float.txt", dtype=float)
    W3_float = np.loadtxt("mnist_cnn_data/W3_float.txt", dtype=float)
    B3_float = np.loadtxt("mnist_cnn_data/B3_float.txt", dtype=float)
    
    img_filename = "mnist_cnn_data/sample_img_0_label_7.txt"
    float_img_filename = "mnist_cnn_data/sample_img_0_float.txt"
    true_label = int(re.search(r'label_(\d+)\.txt', img_filename).group(1))

    x_full = np.loadtxt(img_filename, dtype=np.int64).reshape((1, 28, 28))
    x_float = np.loadtxt(float_img_filename, dtype=float).reshape((1, 28, 28))

    # =========================================================================
    # IDEAL PYTHON CALCULATION (FLOAT NN & INTEGER QUANTIZED)
    # =========================================================================
    print("Calculating Ideal Python & Float NN Scores...")
    f_conv1 = np.maximum(0, python_conv2d_im2col(x_float, W1_float, B1_float, 3, 2, False))
    f_conv2 = np.maximum(0, python_conv2d_im2col(f_conv1, W2_float, B2_float, 3, 2, False))
    f_out = (f_conv2.flatten() @ W3_float) + B3_float
    nn_model_prediction = np.argmax(f_out)

    i_conv1 = np.maximum(0, python_conv2d_im2col(x_full, W1_full, B1_full, 3, 2, True))
    i_conv1_q = i_conv1.astype(np.int64)
    
    i_conv2 = np.maximum(0, python_conv2d_im2col(i_conv1_q, W2_full, B2_full, 3, 2, True))
    i_conv2_q = i_conv2.astype(np.int64)

    i_out_raw = np.dot(i_conv2_q.flatten().astype(object), W3_full.astype(object))
    i_out = np.round(i_out_raw.astype(np.float64) / QAT_SCALE) + B3_full
    ideal_prediction = np.argmax(i_out)

    # =========================================================================
    # HARDWARE SPICE CALCULATION (LAYER BY LAYER)
    # =========================================================================
    hw_conv1 = execute_hardware_conv_layer(W1_full, x_full, 3, 2, SLICES, LRS, HRS, V_READ, "CONV 1")
    hw_conv1 = np.round(hw_conv1 / QAT_SCALE) 
    for c in range(hw_conv1.shape[0]): hw_conv1[c, :, :] += B1_full[c]
    hw_conv1_int = np.maximum(0, hw_conv1).astype(np.int64)

    hw_conv2 = execute_hardware_conv_layer(W2_full, hw_conv1_int, 3, 2, SLICES, LRS, HRS, V_READ, "CONV 2")
    hw_conv2 = np.round(hw_conv2 / QAT_SCALE)
    for c in range(hw_conv2.shape[0]): hw_conv2[c, :, :] += B2_full[c]
    hw_conv2_int = np.maximum(0, hw_conv2).astype(np.int64)

    hw_flattened = hw_conv2_int.flatten()
    print("\n[FC LAYER] Simulating Analog Crossbar...")
    hw_output_raw = execute_hardware_mvm(W3_full, hw_flattened, hw_flattened.shape[0], 10, SLICES, LRS, HRS, V_READ)
    hw_output = np.round(hw_output_raw / QAT_SCALE) + B3_full

    hw_prediction = np.argmax(hw_output)

    # =========================================================================
    # FINAL COMPARISON TABLE
    # =========================================================================
    print("\n" + "=" * 90)
    print("=== MNIST 32-BIT CNN CLASSIFICATION RESULTS ===")
    print("=" * 90)
    print(f"  {'Digit':<8} | {'Hardware Verilog':>20} | {'Ideal Python Score':>19} | {'Float NN Score':>16} | {'Prediction Tag'}")
    print("  " + "-" * 86)

    for col_idx in range(10):
        verilog_val = float(hw_output[col_idx]) / QAT_SCALE
        ideal_val = float(i_out[col_idx]) / QAT_SCALE
        float_val = float(f_out[col_idx])
        
        tags = []
        if col_idx == hw_prediction: tags.append("HW")
        if col_idx == ideal_prediction: tags.append("IDEAL")
        if col_idx == nn_model_prediction: tags.append("FLOAT")
            
        tag_str = f"<-- {', '.join(tags)} PRED" if tags else ""
        print(f"  Digit {col_idx:<2} | {verilog_val:>20.4f} | {ideal_val:>19.4f} | {float_val:>16.4f} | {tag_str}")
        
    print("=" * 90)
    print(f"  TRUE LABEL                  : {true_label}")
    print(f"  FLOAT NN MODEL PREDICTION   : {nn_model_prediction}")
    print(f"  IDEAL PYTHON PREDICTION     : {ideal_prediction}")
    print(f"  HARDWARE VERILOG PREDICTION : {hw_prediction}")
    print("=" * 90)
    
    if hw_prediction == true_label:
        print("\n>> SUCCESS! Hardware CNN crossbar correctly identified the image! <<\n")
    else:
        print("\n>> FAILED. Hardware CNN crossbar misclassified the image. <<\n")
