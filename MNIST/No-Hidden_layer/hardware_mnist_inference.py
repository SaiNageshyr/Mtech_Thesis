#will run only for 8 time because we are using .alter
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
    lines.append("* Resistor Crossbar MVM - NgSPICE .alter Optimization")
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


if __name__ == "__main__":
    ROWS = 784
    COLS = 10
    slices = 8
    
    HRS            = 5.0e19
    LRS            = 5.0e3
    V_READ         = 0.1
    
    SCALING_FACTOR = (1 / V_READ) 
    SF = LRS
    ADC_STEP       = 1 / LRS 
    
    print("\n" + "=" * 60)
    print("  INITIALIZING HARDWARE CROSSBAR FOR MNIST INFERENCE")
    print("=" * 60)

    img_idx = 0
    img_filename = f"mnist_data/sample_img_{img_idx}_label_5.txt"
    float_img_filename = f"mnist_data/sample_img_{img_idx}_float.txt"

    try:
        W_full = np.loadtxt("mnist_data/W_matrix.txt", dtype=int)
        B_full = np.loadtxt("mnist_data/B_vector.txt", dtype=int)
        
        W_float = np.loadtxt("mnist_data/W_float.txt", dtype=float)
        B_float = np.loadtxt("mnist_data/B_float.txt", dtype=float)
        
        x_full = np.loadtxt(img_filename, dtype=int)
        x_float = np.loadtxt(float_img_filename, dtype=float)
        
        true_label = int(re.search(r'label_(\d+)\.txt', img_filename).group(1))
        
        print(f"Successfully loaded model files for Image: {img_filename}")
    except FileNotFoundError:
        print("ERROR: Could not find MNIST data files. Run 'generate_mnist_qat_weights.py' first!")
        exit()

    nn_float_output = (x_float @ W_float) + B_float
    nn_model_prediction = np.argmax(nn_float_output)

    ideal_result = (W_full.T @ x_full) + B_full
    ideal_prediction = np.argmax(ideal_result)

    W_2c = to_2s_complement(W_full, slices)
    x_2c = to_2s_complement(x_full, slices)

    slice_counter = 0
    cycle_slice_data = []
    total_slices = slices * slices
    
    print(f"\nSimulating Analog Crossbar (Running NgSPICE for {total_slices} bit-slices)...")

    # Accelerate I/O by utilizing the Linux RAM disk if available
    ram_disk_dir = "/dev/shm" if os.path.exists("/dev/shm") else None
    temp_dir = tempfile.mkdtemp(dir=ram_disk_dir)

    # -------------------------------------------------------------------------
    # OPTIMIZED LOOP: Run NgSPICE only 'slices' times
    # -------------------------------------------------------------------------
    for w_bit in range(slices):      
        
        W_slice = extract_bit_slice(W_2c, w_bit)
        
        # NgSPICE internally processes all x_bits within this single netlist execution
        nl = build_netlist_with_alter(W_slice, x_2c, ROWS, COLS, LRS, HRS, V_READ, slices)
        log_data = run_ngspice(nl, temp_dir)
        batch_results = parse_currents_batch(log_data, COLS)
        
        # Re-pack the batch data into the standard tuple format for Verilog
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

    print("\nAnalog simulation complete. Sending to Verilog ALU...")

    with open("python_to_verilog.txt", "w") as f_out:
        for combined_shift, raw_currents, slice_sign in cycle_slice_data:
            curr_scaled = raw_currents * SCALING_FACTOR
            for col_idx in range(COLS):
                quantized_mag = adc_quantise(curr_scaled[col_idx], ADC_STEP)
                val_to_write = quantized_mag * slice_sign
                f_out.write(f"{col_idx} {val_to_write} {combined_shift}\n")

    subprocess.run(f"{IVERILOG_CMD} -g2012 -o sim.vvp shift_add_tb_slp.v", shell=True, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(f"{VVP_CMD} sim.vvp", shell=True, check=True, stdout=subprocess.DEVNULL)

    hardware_accumulated = np.zeros(COLS)
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

    final_hardware_output = hardware_accumulated + B_full
    hw_prediction = np.argmax(final_hardware_output)

    print("\n" + "=" * 90)
    print("=== MNIST CLASSIFICATION RESULTS ===")
    print("=" * 90)
    print(f"  {'Digit':<8} | {'Hardware Verilog':>18} | {'Ideal Python Score':>17} | {'Float NN Score':>16} | {'Prediction Tag'}")
    print("  " + "-" * 86)

    for col_idx in range(COLS):
        verilog_val = final_hardware_output[col_idx]
        ideal_val = int(ideal_result[col_idx])
        float_val = float(nn_float_output[col_idx])

        tags = []
        if col_idx == hw_prediction:
            tags.append("HW")
        if col_idx == ideal_prediction:
            tags.append("IDEAL")
        if col_idx == nn_model_prediction:
            tags.append("FLOAT")

        tag_str = f"<-- {', '.join(tags)} PRED" if tags else ""
        print(f"  Digit {col_idx:<2} | {verilog_val:>18.2f} | {ideal_val:>18d} | {float_val:>16.4f} | {tag_str}")

    print("=" * 90)
    print(f"  TRUE LABEL                  : {true_label}")
    print(f"  FLOAT NN MODEL PREDICTION   : {nn_model_prediction}")
    print(f"  IDEAL Python PREDICTION     : {ideal_prediction}")
    print(f"  HARDWARE VERILOG PREDICTION : {hw_prediction}")
    print("=" * 90)

    if hw_prediction == true_label:
        print("\n>> SUCCESS! Hardware crossbar correctly identified the image! <<\n")
    else:
        print("\n>> FAILED. Hardware crossbar misclassified the image. <<\n")
