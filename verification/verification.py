import os
import re
import subprocess
import tempfile
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
#  NgSPICE & Iverilog settings (UPDATED FOR LINUX)
# -----------------------------------------------------------------------------
NGSPICE_CMD   = "ngspice"
NGSPICE_FLAGS = ["-b"]

IVERILOG_CMD  = "iverilog"
VVP_CMD       = "vvp"

# -----------------------------------------------------------------------------
#  Netlist builder
# -----------------------------------------------------------------------------
def build_netlist(W_resistances: np.ndarray, x: np.ndarray, rows: int, cols: int,
                  LRS: float, HRS: float, R_row_wire: float = 1.0, R_col_wire: float = 1.0) -> str:
    lines = []
    lines.append("* Resistor Crossbar MVM - NgSPICE Simulation")
    lines.append(f"* Rows={rows}, Cols={cols}")
    lines.append("")

    lines.append("* -- Input voltage sources ------------------------------")
    for i in range(rows):
        lines.append(f"V_in_{i+1} row_{i+1}_seg_0 0 DC {float(x[i]):.6g}")
    lines.append("")
    lines.append("* -- Row wire segment resistors -------------------------")
    for i in range(rows):
        for k in range(cols - 1):
            node_a = f"row_{i+1}_seg_{k}"
            node_b = f"row_{i+1}_seg_{k+1}"
            lines.append(f"R_row_{i+1}_seg_{k+1}  {node_a}  {node_b}  {R_row_wire:.6g}")
        lines.append("")

    lines.append("* -- Column wire segment resistors ----------------------")
    for j in range(cols):
        for k in range(rows - 1):
            node_a = f"col_{j+1}_seg_{k}"
            node_b = f"col_{j+1}_seg_{k+1}"
            lines.append(f"R_col_{j+1}_seg_{k+1}  {node_a}  {node_b}  {R_col_wire:.6g}")
        lines.append("")
    lines.append("* -- Crossbar Resistors (Direct Connections) ------------")
    for i in range(rows):
        lines.append(f"* row{i+1}")
        for j in range(cols):
            if W_resistances[i][j] == 0:
                lines.append(f"R_r{i+1}c{j+1} row_{i+1}_seg_{j}  col_{j+1}_seg_{i} {HRS:.6g}")
            else:
                lines.append(f"R_r{i+1}c{j+1} row_{i+1}_seg_{j}  col_{j+1}_seg_{i} {LRS:.6g}")
        lines.append("")

    lines.append("* -- Ammeters (0-V sources for current sensing) ---------")
    for j in range(cols):
        lines.append(f"V_ammeter_{j+1} col_{j+1}_seg_{rows-1} 0 DC 0")
    lines.append("")

    lines.append("* -- Analysis --------------------------------------------")
    lines.append(".op")
    lines.append("")

    ammeter_list = " ".join(f"I(V_ammeter_{j+1})" for j in range(cols))
    lines.append("* -- Output ----------------------------------------------")
    lines.append(".control")
    lines.append("run")
    lines.append(f"print {ammeter_list}")
    lines.append(".endc")
    lines.append("")
    lines.append(".end")

    return "\n".join(lines)

# -----------------------------------------------------------------------------
#  NgSPICE runner & parser
# -----------------------------------------------------------------------------
def run_ngspice(netlist_str: str) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sp", delete=False) as f:
        f.write(netlist_str)
        netlist_path = f.name
    log_path = netlist_path + ".log"
    try:
        cmd = [NGSPICE_CMD] + NGSPICE_FLAGS + ["-o", log_path, netlist_path]
        subprocess.run(cmd, capture_output=True, text=True, timeout=None)
        if os.path.exists(log_path):
            with open(log_path, "r") as f_log:
                return f_log.read()
        raise RuntimeError("NgSPICE failed to create log.")
    finally:
        if os.path.exists(netlist_path): os.unlink(netlist_path)
        if os.path.exists(log_path): os.unlink(log_path)

def parse_currents(ngspice_output: str, cols: int) -> np.ndarray:
    currents = np.zeros(cols)
    if "error" in ngspice_output.lower() or "fatal" in ngspice_output.lower():
        raise RuntimeError("NgSPICE failed to simulate.")
    clean_out = ngspice_output.replace("Using SPARSE 1.3 as Direct Linear Solver", "")
    clean_out = re.sub(r'\s+', '', clean_out)

    for j in range(cols):
        pattern = r"i\(v_ammeter_{}\)=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)".format(j+1)
        match = re.search(pattern, clean_out, re.IGNORECASE)
        if match:
            currents[j] = float(match.group(1))
        else:
            raise ValueError(f"Could not find current for Column {j+1}")
    return currents

def print_results(W, x, y, x_bit, w_bit):
    rows, cols = W.shape
    sep = "=" * 60

    print("\nInput voltage vector x (this slice):")
    for i, v in enumerate(x):
        print(f"  row_{i+1:<4}  {v:>14.6g}  ({'1' if v > 0 else '0'})")

    print("\nMatrix W (this slice):")
    for i in range(rows):
        row_str = f"  row_{i+1}:  "
        for j in range(cols):
            row_str += f" {W[i][j]:>4}"
        print(row_str)

# -----------------------------------------------------------------------------
#  Math Helpers
# -----------------------------------------------------------------------------
def extract_bit_slice(matrix: np.ndarray, bit_pos: int) -> np.ndarray:
    return ((matrix >> bit_pos) & 1).astype(int)

def adc_quantise(val: float, step: float) -> float:
    analog_offset = step / 2.0
    # 2. The comparator strictly truncates (floors) the biased signal
    clean_ratio = (val + analog_offset) / step
    return float(np.floor(clean_ratio) * step)

# -----------------------------------------------------------------------------
#  Main Loop
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    
    # -------------------------------------------------------------------------
    # USER CONFIGURATIONS
    # -------------------------------------------------------------------------
    adc_mode = int(input("Enter ADC mode (1: No ADC, 2: With ADC, 3: Both): "))
    slices = int(input("Enter number of slices (e.g., 1 or 4): "))
    iterations = int(input("Enter number of iterations (e.g., 1 or 100): "))
    use_line_res = 0#int(input("Include line resistance? (1: Yes, 0: No): "))
    
    if adc_mode == 1:
        adc_configs = [False]  
    elif adc_mode == 2:
        adc_configs = [True]   
    elif adc_mode == 3:
        adc_configs = [False, True]  
    else:
        print("Invalid choice, defaulting to Both (3).")
        adc_configs = [False, True]

    # -------------------------------------------------------------------------
    # HARDWARE CONSTANTS
    # -------------------------------------------------------------------------
    if use_line_res == 1:
        ROW_WIRE_RESISTANCE = 0.1  
        COL_WIRE_RESISTANCE = 0.1  
    else:
        # 1e-9 acts as an ideal wire (0 ohms) in SPICE without causing topology errors
        ROW_WIRE_RESISTANCE = 1e-9  
        COL_WIRE_RESISTANCE = 1e-9  
        
    LRS            = 5.0e3
    HRS_LIST       = [5.0e19, 5e5, 5e4]  
    V_READ         = 0.1
    SCALING_FACTOR = (1 / V_READ) 
    SF = LRS
    ADC_STEP       = 1/LRS   

    matrix_sizes = [4, 8, 16, 32, 64, 128, 256]
    TEST_PATTERN = "RANDOM" 
    MAX_VAL = (1 << slices)

    print("\n" + "=" * 60)
    res_str = "ON (0.1 ohm)" if use_line_res == 1 else "OFF (Ideal Wires)"
    print(f"=== STARTING VERILOG CO-SIMULATION (PATTERN: {TEST_PATTERN} | LINE RES: {res_str}) ===")
    print("=" * 60)
    
    for hrs in HRS_LIST:
        
        plot_sizes = []
        plot_avg_python = []
        plot_avg_ngspice = {False: [], True: []}
        plot_avg_error = {False: [], True: []}
        
        # Trackers for graph display
        captured_texts = []

        for size in matrix_sizes:
            ROWS = size
            COLS = 1 
            
            print(f"\nProcessing Matrix Size: {ROWS}x{COLS} ({iterations} iterations)...")
            
            if TEST_PATTERN == "ZEROS":
                x_full = np.ones(ROWS, dtype=int) 
            elif TEST_PATTERN == "ONES":
                x_full = np.ones(ROWS, dtype=int)
            else:
                x_full = np.random.randint(0, MAX_VAL, size=ROWS)
                
            cycle_python_vals = []
            cycle_ngspice_vals = {False: [], True: []}
            cycle_errors = {False: [], True: []}

            for cycle in range(iterations):
                if iterations > 1 and (cycle + 1) % 10 == 0:
                    print(f"  -> Cycle {cycle+1}/{iterations}...")

                if TEST_PATTERN == "ZEROS":
                    W_full = np.zeros((ROWS, COLS), dtype=int)
                elif TEST_PATTERN == "ONES":
                    W_full = np.ones((ROWS, COLS), dtype=int)
                else:
                    W_full = np.random.randint(0, MAX_VAL, size=(ROWS, COLS))

                # Capture the matrices for the graph if iterations == 1 AND size <= 32
                if iterations == 1 and size <= 32:
                    x_str = np.array2string(x_full, threshold=12, edgeitems=3)
                    w_str = np.array2string(W_full.flatten(), threshold=12, edgeitems=3)
                    captured_texts.append(f"Size {size:<2}x1 | x: {x_str} | W^T: {w_str}")

                pure_python_result = W_full.T @ x_full
                cycle_python_vals.append(np.mean(pure_python_result))
                
                if iterations == 1:
                    print(f"W = {W_full}")
                    print(f"x = {x_full}")
                try:
                    cycle_slice_data = [] 
                    
                    for x_bit in range(slices):          
                        for w_bit in range(slices):  
                            x_slice = extract_bit_slice(x_full, x_bit)
                            W_slice = extract_bit_slice(W_full, w_bit)
                            x_volts = x_slice.astype(float) * V_READ
                            
                            netlist = build_netlist(W_slice, x_volts, ROWS, COLS, LRS, hrs, R_row_wire=ROW_WIRE_RESISTANCE, R_col_wire=COL_WIRE_RESISTANCE)
                            raw_output = run_ngspice(netlist)
                            currents = parse_currents(raw_output, COLS)
                            
                            if iterations == 1:
                                print_results(W_slice, x_slice, currents, x_bit, w_bit)
                                
                            combined_shift = x_bit + w_bit
                            cycle_slice_data.append((combined_shift, currents))

                    for use_adc in adc_configs:
                        with open("python_to_verilog.txt", "w") as f_out:
                            for combined_shift, currents in cycle_slice_data:
                                curr_scaled = currents * SCALING_FACTOR
                                
                                for j in range(COLS):
                                    if use_adc:
                                        val_to_write = adc_quantise(curr_scaled[j], ADC_STEP)
                                    else:
                                        val_to_write = curr_scaled[j] 
                                        
                                    f_out.write(f"{j} {val_to_write} {combined_shift}\n")
                                    
                                    if iterations == 1:
                                        print(f"{currents[j]:>16e}  {curr_scaled[j]:>16e}  {val_to_write:>16e}")

                        subprocess.run(f"{IVERILOG_CMD} -g2012 -o sim.vvp shift_add_tb.v", shell=True, check=True, stdout=subprocess.DEVNULL)
                        subprocess.run(f"{VVP_CMD} sim.vvp", shell=True, check=True, stdout=subprocess.DEVNULL)

                        hardware_accumulated = np.zeros(COLS)
                        if os.path.exists("verilog_to_python.txt"):
                            with open("verilog_to_python.txt", "r") as f_in:
                                for line in f_in:
                                    if line.strip():
                                        parts = line.split()
                                        col_idx = int(parts[0])
                                        hardware_accumulated[col_idx] = float(parts[1]) * SF
                        else:
                            raise RuntimeError("Verilog output file not found!")

                        # -------------------------------------------------------------
                        # UPDATED Error Calculation Logic
                        # -------------------------------------------------------------
                        total_error = 0.0
                        valid_cols = 0
                        
                        for j in range(COLS):
                            a_py = float(pure_python_result[j])
                            a_hw = float(hardware_accumulated[j])
                            
                            if TEST_PATTERN == "ZEROS":
                                # Calculate error even if python is 0, add epsilon to prevent crash
                                col_error = abs((a_hw - a_py) / (a_py + 1e-30))
                                total_error += col_error
                                valid_cols += 1
                            else:
                                # Skip error calculation if Python value accidentally hit exactly 0
                                if a_py != 0:
                                    col_error = abs((a_hw - a_py) / a_py)
                                    total_error += col_error
                                    valid_cols += 1
                        
                        cycle_ngspice_vals[use_adc].append(np.mean(hardware_accumulated))
                        
                        if valid_cols > 0:
                            cycle_errors[use_adc].append(total_error / valid_cols)
                        else:
                            cycle_errors[use_adc].append(0.0)
                        
                except Exception as e:
                    print(f"  Error processing size {size} on cycle {cycle}: {e}")
                    break

            if len(cycle_python_vals) > 0:
                avg_python = np.mean(cycle_python_vals)
                plot_sizes.append(size)
                plot_avg_python.append(avg_python)
                print(f"  Avg Python Value:  {avg_python:.6f}")
                
                for use_adc in adc_configs:
                    avg_ngspice = np.mean(cycle_ngspice_vals[use_adc])
                    avg_error = np.mean(cycle_errors[use_adc])
                    plot_avg_ngspice[use_adc].append(avg_ngspice)
                    plot_avg_error[use_adc].append(avg_error)
                    
                    mode_str = "With ADC" if use_adc else "No ADC  "
                    print(f"  [{mode_str}] Avg Verilog: {avg_ngspice:.6f} | Error: {avg_error:.6e}")

        # -------------------------------------------------------------------------
        # Generate the Graphs (Memory only, Display Deferred)
        # -------------------------------------------------------------------------
        if plot_sizes:
            print(f"\nGenerating Graphs for HRS = {hrs} (in background)...")
            
            # --- GRAPH 1: Python vs Verilog Values ---
            plt.figure(figsize=(10, 6.5))
            plot_avg_python_safe = [p + 1e-30 if p == 0 else p for p in plot_avg_python]
            plt.plot(plot_sizes, plot_avg_python_safe, marker='o', linestyle='-', color='g', linewidth=2, markersize=8, label='Ideal Python Values')
            
            if False in adc_configs:
                hw_safe = [p + 1e-30 if p == 0 else p for p in plot_avg_ngspice[False]]
                plt.plot(plot_sizes, hw_safe, marker='s', linestyle='--', color='orange', linewidth=2, markersize=8, label='Hardware (No ADC)')
                
            if True in adc_configs:
                hw_safe = [p + 1e-30 if p == 0 else p for p in plot_avg_ngspice[True]]
                plt.plot(plot_sizes, hw_safe, marker='d', linestyle='--', color='blue', linewidth=2, markersize=8, label='Hardware (With ADC)')
            
            plt.title(f'Ideal vs Real Hardware Results (HRS = {hrs})', fontsize=14, fontweight='bold')
            plt.xlabel('Matrix Size (N x N)', fontsize=12)
            plt.ylabel('Digital Output Value', fontsize=12)
            
            plt.xscale('log', base=2)
            if TEST_PATTERN == "ZEROS":
                plt.yscale('log')
                
            plt.xticks(plot_sizes, [f"{s}" for s in plot_sizes])
            plt.grid(True, which="both", linestyle="--", linewidth=0.5)
            plt.legend(loc="upper left")
            
            # Format layout based on iterations
            if iterations == 1 and captured_texts:
                text_content = "Inputs for lower sizes (W flattened to W^T for compact display):\n" + "\n".join(captured_texts)
                plt.figtext(0.5, 0.98, text_content, ha='center', va='top', fontsize=9, family='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
                plt.subplots_adjust(top=0.75, bottom=0.1) 
            else:
                plt.tight_layout()

            # --- GRAPH 2: Calculated Error ---
            plt.figure(figsize=(10, 6.5))
            
            if False in adc_configs:
                if TEST_PATTERN == "ZEROS":
                    err_safe = [max(e, 1e-30) for e in plot_avg_error[False]]
                else:
                    err_safe = plot_avg_error[False]
                plt.plot(plot_sizes, err_safe, marker='^', linestyle='-', color='orange', linewidth=2, markersize=8, label='Relative Error (No ADC)')
                
            if True in adc_configs:
                if TEST_PATTERN == "ZEROS":
                    err_safe = [max(e, 1e-30) for e in plot_avg_error[True]]
                else:
                    err_safe = plot_avg_error[True]
                plt.plot(plot_sizes, err_safe, marker='v', linestyle='-', color='red', linewidth=2, markersize=8, label='Relative Error (With ADC)')
            
            plt.title(f'Calculated Error vs Matrix Size (HRS = {hrs})', fontsize=14, fontweight='bold')
            plt.xlabel('Matrix Size (N x N)', fontsize=12)
            plt.ylabel('Calculated Error Ratio', fontsize=12)
            
            plt.xscale('log', base=2)
            
            # -------------------------------------------------------------
            # UPDATED Y-Axis Scale Logic
            # -------------------------------------------------------------
            if TEST_PATTERN == "ZEROS":
                plt.yscale('log') 
            else:
                plt.yscale('linear')
                
            plt.xticks(plot_sizes, [f"{s}" for s in plot_sizes])
            plt.grid(True, which="both", linestyle="--", linewidth=0.5)
            plt.legend(loc="upper left")
            
            # Apply same formatting to second graph
            if iterations == 1 and captured_texts:
                plt.figtext(0.5, 0.98, text_content, ha='center', va='top', fontsize=9, family='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
                plt.subplots_adjust(top=0.75, bottom=0.1)
            else:
                plt.tight_layout()

    # -------------------------------------------------------------------------
    # Display All Graphs Simultaneously
    # -------------------------------------------------------------------------
    print("\nSimulation complete! Opening all graphs simultaneously...")
    plt.show()
