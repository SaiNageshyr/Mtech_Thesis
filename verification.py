import os
import re
import subprocess
import tempfile
import numpy as np
import matplotlib.pyplot as plt
# -----------------------------------------------------------------------------
#  NgSPICE settings
# -----------------------------------------------------------------------------
NGSPICE_CMD   = r"D:\CouchEd_projects\CouchEd\ngspice-42_64\Spice64\bin\ngspice_con.exe"
NGSPICE_FLAGS = ["-b"]

# -----------------------------------------------------------------------------
#  Netlist builder
# -----------------------------------------------------------------------------
def build_netlist(W_resistances: np.ndarray, x: np.ndarray, rows: int, cols: int,
                  LRS: float, HRS: float,
                  R_row_wire: float = 1.0,
                  R_col_wire: float = 1.0) -> str:

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

    lines.append("* -- Crossbar Resistors ----------------------------------")
    for i in range(rows):
        lines.append(f"* row{i+1}")
        for j in range(cols):
            if W_resistances[i][j] == 0:
                lines.append(f"R_r{i+1}c{j+1} row_{i+1}_seg_{j}  col_{j+1}_seg_{i} {HRS:.6g}")
            else:
                lines.append(f"R_r{i+1}c{j+1}  row_{i+1}_seg_{j}  col_{j+1}_seg_{i}  {LRS:.6g}")
        lines.append("")

    lines.append("* -- Ammeters (0-V sources for current sensing) ---------")
    for j in range(cols):
        last_col_node = f"col_{j+1}_seg_{rows-1}"
        lines.append(f"V_ammeter_{j+1}  {last_col_node}  0  DC 0")
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
#  NgSPICE runner
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
        else:
            raise RuntimeError("NgSPICE failed to create an output log file.")

    finally:
        if os.path.exists(netlist_path):
            os.unlink(netlist_path)
        if os.path.exists(log_path):
            os.unlink(log_path)

# -----------------------------------------------------------------------------
#  Current parser
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#  Current parser (Updated for thread-safety)
# -----------------------------------------------------------------------------
def parse_currents(ngspice_output: str, cols: int) -> np.ndarray:
    currents = np.zeros(cols)

    if "error" in ngspice_output.lower() or "fatal" in ngspice_output.lower():
        print("\n=== NGSPICE FATAL ERROR DETECTED ===")
        print(ngspice_output)
        print("====================================\n")
        raise RuntimeError("NgSPICE failed to simulate the circuit.")
    # 1. Remove the interrupting NgSPICE solver message
    clean_out = ngspice_output.replace("Using SPARSE 1.3 as Direct Linear Solver", "")
    
    # 2. Strip ALL whitespace and newlines. 
    # This forces broken words to snap back together (e.g., "i(v_a \n mmeter_12)" -> "i(v_ammeter_12)")
    clean_out = re.sub(r'\s+', '', clean_out)

    for j in range(cols):
        # 3. Because all spaces are gone, our regex simply looks for "i(v_ammeter_X)=Y"
        pattern = r"i\(v_ammeter_{}\)=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)".format(j+1)
        
        # Using re.IGNORECASE just in case NgSPICE outputs uppercase 'I' or 'V'
        match = re.search(pattern, clean_out, re.IGNORECASE)
        
        if match:
            currents[j] = float(match.group(1))
        else:
            print("\n=== RAW NGSPICE OUTPUT (DEBUG) ===")
            print(ngspice_output)
            print("==================================\n")
            raise ValueError(f"Could not find current for Column {j+1} in output.")

    return currents

# -----------------------------------------------------------------------------
#  Results Printer
# -----------------------------------------------------------------------------
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

    print("\nNgSPICE output currents:")
    print(f"  {'Column':<8}  {'Current (A)':>16}  {'digit_val':>10}")
    print("  " + "-" * 40)
    for j, current in enumerate(y):
        digit = round(current * 50000)
        print(f"  col_{j+1:<4}  {current:>16.6e}  {digit:>10}")
    print(sep + "\n")

def python_cal(W: np.ndarray, x: np.ndarray, rows: int, cols: int, HRS: float, LRS: float) -> np.ndarray:
    output_I = np.zeros(cols)
    for k in range(cols):
        out = 0.0  
        for i in range(rows):
            if W[i][k] == 0:
                out = out + (x[i] / HRS)
            else:
                out = out + (x[i] / LRS)
        output_I[k] = out
    print("\n" + "=" * 40)
    print("  Currents calculated using Python (Ideal)")
    print("=" * 40)
    for k in range(cols):
        print(f"  current_in_col_{k+1} = {output_I[k]:.6e} A")

# -----------------------------------------------------------------------------
#  Python ideal verification
# -----------------------------------------------------------------------------
def python_mvm(W_full: np.ndarray, x_full: np.ndarray, cols: int):
    result = W_full.T @ x_full          # shape: (cols,)
    print("\n" + "=" * 50)
    print("  IDEAL Python Matrix-Vector Multiplication")
    print("=" * 50)
    print(f"\n  W =\n{W_full}")
    print(f"\n  x = {x_full}")
    print(f"\n  W^T · x =")
    for j in range(cols):
        print(f"    col_{j+1} = {result[j]}")
    print("=" * 50 + "\n")
    return result

# -----------------------------------------------------------------------------
#  Bit-slice extractor
# -----------------------------------------------------------------------------
def extract_bit_slice(matrix: np.ndarray, bit_pos: int) -> np.ndarray:
    """Extract a single bit plane from an integer matrix."""
    return ((matrix >> bit_pos) & 1).astype(int)

# -----------------------------------------------------------------------------
#  Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # User inputs
    # -------------------------------------------------------------------------
    slices = 1#int(input("Enter number of slices: "))

    # SET WIRE RESISTANCE TO 0
    ROW_WIRE_RESISTANCE = 0.0
    COL_WIRE_RESISTANCE = 0.0
    
    LRS            = 5.0e3
    SCALING_FACTOR = 50000
    V_READ         = 0.1

    MAX_VAL = (1 << slices)   # 2^slices
    matrix_sizes = [4, 8, 16, 32, 64, 128]
    
    # --- GRAPHING ARRAYS (Now storing Means and Standard Deviations) ---
    plot_sizes = []
    
    plot_mean_5e19 = []; plot_std_5e19 = []
    plot_mean_5e5  = []; plot_std_5e5  = []
    plot_mean_5e4  = []; plot_std_5e4  = []

    for size in matrix_sizes:
        ROWS = size
        COLS = size
        
        print("\n" + "=" * 60)
        print(f"=== STARTING MATRIX SIZE: {ROWS}x{COLS} (100 Iterations) ===")
        print("=" * 60)
        
        # Lists to hold the error of each individual cycle (100 items total)
        cycle_errors_5e19 = []
        cycle_errors_5e5  = []
        cycle_errors_5e4  = []

        for i in range(100):
            if (i + 1) % 10 == 0:
                print(f"  -> Processing iteration {i+1}/100...")
            
            W_full = np.random.randint(0, MAX_VAL, size=(ROWS, COLS))
            x_full = np.random.randint(0, MAX_VAL, size=ROWS)
            
            pure_python_result = W_full.T @ x_full
            
            # Open three separate files for Verilog inputs
            with open("py_to_v_5e19.txt", "w") as f_5e19, open("py_to_v_5e5.txt", "w") as f_5e5, open("py_to_v_5e4.txt", "w") as f_5e4:
                for x_bit in range(slices):          
                    for w_bit in range(slices):      
                        combined_shift = x_bit + w_bit

                        x_slice = extract_bit_slice(x_full, x_bit).astype(float) * V_READ
                        W_slice = extract_bit_slice(W_full, w_bit)

                        try:
                            # 1. Base Ideal (5e19)
                            nl_5e19 = build_netlist(W_slice, x_slice, ROWS, COLS, LRS, 5.0e19, ROW_WIRE_RESISTANCE, COL_WIRE_RESISTANCE)
                            curr_5e19 = parse_currents(run_ngspice(nl_5e19), COLS)
                            
                            # 2. Test 1 (5e5)
                            nl_5e5 = build_netlist(W_slice, x_slice, ROWS, COLS, LRS, 5.0e5, ROW_WIRE_RESISTANCE, COL_WIRE_RESISTANCE)
                            curr_5e5 = parse_currents(run_ngspice(nl_5e5), COLS)

                            # 3. Test 2 (5e4) - NEW
                            nl_5e4 = build_netlist(W_slice, x_slice, ROWS, COLS, LRS, 5.0e4, ROW_WIRE_RESISTANCE, COL_WIRE_RESISTANCE)
                            curr_5e4 = parse_currents(run_ngspice(nl_5e4), COLS)
                            
                            # Write to respective files
                            for c in range(COLS):
                                f_5e19.write(f"{c} {curr_5e19[c] * SCALING_FACTOR:.6f} {combined_shift}\n")
                                f_5e5.write(f"{c} {curr_5e5[c] * SCALING_FACTOR:.6f} {combined_shift}\n")
                                f_5e4.write(f"{c} {curr_5e4[c] * SCALING_FACTOR:.6f} {combined_shift}\n")
                                
                        except Exception as e:
                            print(f"  Error in SPICE: {e}")

            # -------------------------------------------------------------------------
            # Hand off to Verilog (Shift-and-Add) for all 3 cases
            # (Note: You must ensure your shift_add_tb.v is set to read these 3 specific filenames!)
            # -------------------------------------------------------------------------
            try:
                subprocess.run(r"C:\iverilog\bin\iverilog -g2012 -o sim_5e19.vvp shift_add_tb_5e19.v", shell=True, check=True, stdout=subprocess.DEVNULL)
                subprocess.run(r"C:\iverilog\bin\vvp sim_5e19.vvp", shell=True, check=True, stdout=subprocess.DEVNULL)

                subprocess.run(r"C:\iverilog\bin\iverilog -g2012 -o sim_5e5.vvp shift_add_tb_5e5.v", shell=True, check=True, stdout=subprocess.DEVNULL)
                subprocess.run(r"C:\iverilog\bin\vvp sim_5e5.vvp", shell=True, check=True, stdout=subprocess.DEVNULL)

                subprocess.run(r"C:\iverilog\bin\iverilog -g2012 -o sim_5e4.vvp shift_add_tb_5e4.v", shell=True, check=True, stdout=subprocess.DEVNULL)
                subprocess.run(r"C:\iverilog\bin\vvp sim_5e4.vvp", shell=True, check=True, stdout=subprocess.DEVNULL)

            except subprocess.CalledProcessError:
                print("Error: Verilog compilation failed.")

            # -------------------------------------------------------------------------
            # Calculate Errors for this specific cycle
            # -------------------------------------------------------------------------
            if os.path.exists("v_out_5e19.txt") and os.path.exists("v_out_5e5.txt") and os.path.exists("v_out_5e4.txt"):
                
                err_sum_5e19 = 0.0
                err_sum_5e5  = 0.0
                err_sum_5e4  = 0.0
                valid_cols = 0
                
                with open("v_out_5e19.txt", "r") as f1, open("v_out_5e5.txt", "r") as f2, open("v_out_5e4.txt", "r") as f3:
                    for l1, l2, l3 in zip(f1, f2, f3):
                        if l1.strip():
                            p1, p2, p3 = l1.split(), l2.split(), l3.split()
                            col_idx = int(p1[0])
                            
                            a_j_pure = float(pure_python_result[col_idx])
                            a_j_5e19 = float(p1[1])
                            a_j_5e5  = float(p2[1])
                            a_j_5e4  = float(p3[1])
                            
                            if a_j_pure != 0:
                                err_sum_5e19 += abs(a_j_pure - a_j_5e19) / abs(a_j_pure)
                                err_sum_5e5  += abs(a_j_pure - a_j_5e5) / abs(a_j_pure)
                                err_sum_5e4  += abs(a_j_pure - a_j_5e4) / abs(a_j_pure)
                                valid_cols += 1
                
                # Append this cycle's average error to the tracking lists
                if valid_cols > 0:
                    cycle_errors_5e19.append(err_sum_5e19 / valid_cols)
                    cycle_errors_5e5.append(err_sum_5e5 / valid_cols)
                    cycle_errors_5e4.append(err_sum_5e4 / valid_cols)

        # -------------------------------------------------------------------------
        # Calculate Mean and Standard Deviation over the 100 Runs
        # -------------------------------------------------------------------------
        if len(cycle_errors_5e5) > 0:
            
            # Numpy computes the Mean and Std Dev effortlessly!
            mean_5e19, std_5e19 = np.mean(cycle_errors_5e19), np.std(cycle_errors_5e19)
            mean_5e5,  std_5e5  = np.mean(cycle_errors_5e5),  np.std(cycle_errors_5e5)
            mean_5e4,  std_5e4  = np.mean(cycle_errors_5e4),  np.std(cycle_errors_5e4)
            
            print(f"\n=== RESULTS FOR SIZE {ROWS}x{COLS} (100 Cycles) ===")
            print(f"  5e19 -> Mean Error: {mean_5e19:.6f} | Std Dev: {std_5e19:.6f}")
            print(f"  5e5  -> Mean Error: {mean_5e5:.6f}  | Std Dev: {std_5e5:.6f}")
            print(f"  5e4  -> Mean Error: {mean_5e4:.6f}  | Std Dev: {std_5e4:.6f}")
            print("========================================================\n")
            
            plot_sizes.append(size)
            plot_mean_5e19.append(mean_5e19); plot_std_5e19.append(std_5e19)
            plot_mean_5e5.append(mean_5e5);   plot_std_5e5.append(std_5e5)
            plot_mean_5e4.append(mean_5e4);   plot_std_5e4.append(std_5e4)

    # -------------------------------------------------------------------------
    # Generate the Graph with Standard Deviation Error Bars
    # -------------------------------------------------------------------------
    if plot_sizes:
        print("Generating Error Analysis Graph...")
        
        plt.figure(figsize=(10, 6))
        
        # Using errorbar() to plot the mean and display the standard deviation as vertical whiskers
        plt.errorbar(plot_sizes, plot_mean_5e19, yerr=plot_std_5e19, marker='o', color='g', label='HRS = 5e19 (Ideal)', capsize=5)
        plt.errorbar(plot_sizes, plot_mean_5e5,  yerr=plot_std_5e5,  marker='s', color='b', label='HRS = 5e5', capsize=5)
        plt.errorbar(plot_sizes, plot_mean_5e4,  yerr=plot_std_5e4,  marker='^', color='r', label='HRS = 5e4 (Worst Leakage)', capsize=5)
        
        plt.title('Leakage Error vs Matrix Size with Standard Deviation (No Wires)', fontsize=14, fontweight='bold')
        plt.xlabel('Matrix Size (N x N)', fontsize=12)
        plt.ylabel('Relative Error (Ratio)', fontsize=12)
        
        plt.xscale('log',base=2)
        plt.yscale('log')
        plt.xticks(plot_sizes, [f"{s}" for s in plot_sizes])
        
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.legend(loc="upper left")
        
        plt.tight_layout()
        plt.show()
