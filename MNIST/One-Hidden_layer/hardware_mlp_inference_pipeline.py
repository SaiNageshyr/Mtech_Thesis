#making the changes to lanuch the ngspice only slice number of times
import os
import re
import subprocess
import tempfile
import numpy as np

# -----------------------------------------------------------------------------
#  NgSPICE & Iverilog settings (LINUX COMPATIBLE)
# -----------------------------------------------------------------------------
NGSPICE_CMD   = "ngspice"
NGSPICE_FLAGS = ["-b", "-q"]
IVERILOG_CMD  = "iverilog"
VVP_CMD       = "vvp"

# -----------------------------------------------------------------------------
#  Netlist builder (Optimized with .alter and Ideal Wires)
# -----------------------------------------------------------------------------
def build_netlist_with_alter(W_slice: np.ndarray, x_2c: np.ndarray, rows: int, cols: int,
                             LRS: float, HRS: float, V_READ: float, slices: int) -> str:
    lines = []
    lines.append("* Resistor Crossbar MVM - NgSPICE .alter Optimization (Ideal Wires)")
    lines.append(f"* Rows={rows}, Cols={cols}\n")

    # Initialize standard voltage sources (Defaulting to x_bit = 0)
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
    
    # -------------------------------------------------------------------------
    # INTERNAL SPICE LOOP (Replaces massive OS process spawning)
    # -------------------------------------------------------------------------
    lines.append("* -- Control Block (Looping through Input Voltages) --")
    lines.append(".control")
    
    for x_bit in range(slices):
        # Print a marker so Python knows which bit-slice these currents belong to
        lines.append(f"echo 'X_BIT_MARKER: {x_bit}'")
        
        x_slice = extract_bit_slice(x_2c, x_bit)
        x_volts = x_slice.astype(float) * V_READ
        
        # Instantly alter all row voltage sources in RAM
        for i in range(rows):
            lines.append(f"alter V_in_{i+1} dc = {float(x_volts[i]):.6g}")
            
        lines.append("op")
        lines.append(f"print {ammeter_list}")
        
    lines.append(".endc")
    lines.append("\n.end")
    
    return "\n".join(lines)

# -----------------------------------------------------------------------------
#  NgSPICE runner & batch parser
# -----------------------------------------------------------------------------
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
    
    # Split the output log using the marker we echoed in the .control block
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

# -----------------------------------------------------------------------------
#  Math Helpers
# -----------------------------------------------------------------------------
def extract_bit_slice(matrix: np.ndarray, bit_pos: int) -> np.ndarray:
    return ((matrix >> bit_pos) & 1).astype(int)

def adc_quantise(val: float, step: float) -> float:
    clean_ratio = np.round(val / step, decimals=4)
    return float(np.round(clean_ratio) * step)

def to_2s_complement(matrix: np.ndarray, bits: int) -> np.ndarray:
    return np.where(matrix < 0, (1 << bits) + matrix, matrix)

# -----------------------------------------------------------------------------
#  Hardware Layer Execution Function
# -----------------------------------------------------------------------------
def execute_hardware_layer(W_layer: np.ndarray, x_input: np.ndarray, 
                           rows: int, cols: int, slices: int, 
                           LRS: float, HRS: float, V_READ: float, 
                           layer_name: str) -> np.ndarray:
    
    total_slices = slices * slices
    print(f"\n[{layer_name}] Simulating Analog Crossbar ({rows}x{cols})...")
    print(f"[{layer_name}] Running NgSPICE for {total_slices} bit-slices...")

    W_2c = to_2s_complement(W_layer, slices)
    x_2c = to_2s_complement(x_input, slices)

    SCALING_FACTOR = (1 / V_READ)
    SF = LRS
    ADC_STEP = 1 / LRS 

    slice_counter = 0
    cycle_slice_data = []
    
    # Automatically use the RAM disk (/dev/shm) if running on Linux
    ram_disk_dir = "/dev/shm" if os.path.exists("/dev/shm") else None
    temp_dir = tempfile.mkdtemp(dir=ram_disk_dir)

    # 1. Analog SPICE Simulation
    for w_bit in range(slices):      
        
        W_slice = extract_bit_slice(W_2c, w_bit)
        
        # Build the master netlist containing all 32 X_bit voltage alterations
        nl = build_netlist_with_alter(W_slice, x_2c, rows, cols, LRS, HRS, V_READ, slices)
        
        # Run NgSPICE (Executes all 32 voltage slices internally)
        log_data = run_ngspice(nl, temp_dir)
        
        # Parse the batch results into a dictionary
        batch_results = parse_currents_batch(log_data, cols)
        
        # 2. Re-pack the batch data into the standard tuple format for Verilog
        for x_bit in range(slices):
            slice_counter += 1
            combined_shift = x_bit + w_bit

            is_x_msb = (x_bit == slices - 1)
            is_w_msb = (w_bit == slices - 1)
            slice_sign = -1.0 if (is_x_msb != is_w_msb) else 1.0  
            
            raw_currents = batch_results[x_bit]
            cycle_slice_data.append((combined_shift, raw_currents, slice_sign))
            
            percent = (slice_counter / total_slices) * 100
            print(f"  -> Processed slice {slice_counter}/{total_slices} ({percent:.1f}%)", end='\r')

    # Cleanup the temp directory after simulation finishes
    os.rmdir(temp_dir)

    print(f"\n[{layer_name}] Analog SPICE complete. Running Verilog ALU Shift-Adder...")

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
    else:
        raise RuntimeError("Verilog output file verilog_to_python.txt not found!")
    
    return hardware_accumulated

# -----------------------------------------------------------------------------
#  Main Loop
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    INPUT_SIZE = 784
    HIDDEN_SIZE = 64
    OUTPUT_SIZE = 10
    SLICES = 32 
    
    HRS            = 5.0e19
    LRS            = 5.0e3
    V_READ         = 0.1
    
    print("\n" + "=" * 64)
    print("  INITIALIZING HARDWARE MULTI-LAYER PERCEPTRON (MLP)")
    print("  Architecture: 784 -> 64 -> 10 | Precision: 32-bit")
    print("=" * 64)

    # 1. Load integer quantized weights & biases
    W1_full = np.loadtxt("mnist_mlp_data/W1_matrix.txt", dtype=int)
    B1_full = np.loadtxt("mnist_mlp_data/B1_vector.txt", dtype=int)
    W2_full = np.loadtxt("mnist_mlp_data/W2_matrix.txt", dtype=int)
    B2_full = np.loadtxt("mnist_mlp_data/B2_vector.txt", dtype=int)
    
    # 2. Load original FP32 float weights & biases
    W1_float = np.loadtxt("mnist_mlp_data/W1_float.txt", dtype=float)
    B1_float = np.loadtxt("mnist_mlp_data/B1_float.txt", dtype=float)
    W2_float = np.loadtxt("mnist_mlp_data/W2_float.txt", dtype=float)
    B2_float = np.loadtxt("mnist_mlp_data/B2_float.txt", dtype=float)
    
    img_filename = "mnist_mlp_data/sample_img_0_label_5.txt"
    float_img_filename = "mnist_mlp_data/sample_img_0_float.txt"
    
    x_full = np.loadtxt(img_filename, dtype=int)
    x_float = np.loadtxt(float_img_filename, dtype=float)
    true_label = int(re.search(r'label_(\d+)\.txt', img_filename).group(1))

    nn_float_hidden = np.maximum(0, (x_float @ W1_float) + B1_float)
    nn_float_output = (nn_float_hidden @ W2_float) + B2_float
    nn_model_prediction = np.argmax(nn_float_output)

    ideal_hidden = (W1_full.T @ x_full) + B1_full
    ideal_hidden_relu = np.maximum(0, ideal_hidden)

    hw_hidden = execute_hardware_layer(W1_full, x_full, INPUT_SIZE, HIDDEN_SIZE, SLICES, 
                                       LRS, HRS, V_READ, "LAYER 1")
    hw_hidden += B1_full

    print("\n[DIGITAL] Applying ReLU & Re-Quantizing Activations for Layer 2...")
    hw_hidden_relu = np.maximum(0, hw_hidden)
    
    max_act = np.max(hw_hidden_relu)
    if max_act > 0:
        hw_hidden_relu_int = np.round((hw_hidden_relu / max_act) * 255).astype(int)
        ideal_hidden_relu_scaled = np.round((ideal_hidden_relu / max_act) * 255).astype(int)
    else:
        hw_hidden_relu_int = np.zeros_like(hw_hidden_relu, dtype=int)
        ideal_hidden_relu_scaled = np.zeros_like(ideal_hidden_relu, dtype=int)

    ideal_output_scaled = (W2_full.T @ ideal_hidden_relu_scaled) + B2_full

    hw_output = execute_hardware_layer(W2_full, hw_hidden_relu_int, HIDDEN_SIZE, OUTPUT_SIZE, SLICES, 
                                       LRS, HRS, V_READ, "LAYER 2")
    hw_output += B2_full

    hw_prediction = np.argmax(hw_output)
    ideal_prediction = np.argmax(ideal_output_scaled)

    print("\n" + "=" * 90)
    print("=== MNIST MLP CLASSIFICATION RESULTS ===")
    print("=" * 90)
    print(f"  {'Digit':<8} | {'Hardware Verilog':>20} | {'Ideal Quant Score':>20} | {'Float NN Score':>16} | {'Prediction Tag'}")
    print("  " + "-" * 86)

    for col_idx in range(OUTPUT_SIZE):
        verilog_val = hw_output[col_idx]
        ideal_val = int(ideal_output_scaled[col_idx])
        float_val = float(nn_float_output[col_idx])
        
        tags = []
        if col_idx == hw_prediction:
            tags.append("HW")
        if col_idx == ideal_prediction:
            tags.append("IDEAL")
        if col_idx == nn_model_prediction:
            tags.append("FLOAT")
            
        tag_str = f"<-- {', '.join(tags)} PRED" if tags else ""
        print(f"  Digit {col_idx:<2} | {verilog_val:>20.2f} | {ideal_val:>20d} | {float_val:>16.4f} | {tag_str}")
        
    print("=" * 90)
    print(f"  TRUE LABEL                  : {true_label}")
    print(f"  FLOAT NN MODEL PREDICTION   : {nn_model_prediction}")
    print(f"  IDEAL QUANTIZED PREDICTION  : {ideal_prediction}")
    print(f"  HARDWARE VERILOG PREDICTION : {hw_prediction}")
    print("=" * 90)
    
    if hw_prediction == true_label:
        print("\n>> SUCCESS! Hardware crossbar correctly identified the image! <<\n")
    else:
        print("\n>> FAILED. Hardware crossbar misclassified the image. <<\n")
