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
    slices = int(input("Enter number of slices: "))

    ROW_WIRE_RESISTANCE = 0.1
    COL_WIRE_RESISTANCE = 0.1
    HRS            = 5.0e5
    LRS            = 5.0e3
    SCALING_FACTOR = 50000
    V_READ         = 0.1

    MAX_VAL = (1 << slices)   # 2^slices
    total_slices = slices * slices

    matrix_sizes = [4, 8, 16,]
    
    # --- GRAPHING ARRAYS ---
    plot_sizes = []
    plot_errors = []

    for size in matrix_sizes:
        ROWS = size
        COLS = size
        
        print("\n" + "=" * 60)
        print(f"=== STARTING MATRIX SIZE: {ROWS}x{COLS} (100 Iterations) ===")
        print("=" * 60)
        
        cumulative_error_100_runs = 0.0
        total_cycles_evaluated = 0

        for i in range(100):
            if (i + 1) % 10 == 0:
                print(f"  -> Processing iteration {i+1}/100...")
            
            slice_counter = 0 
            
            W_full = np.random.randint(0, MAX_VAL, size=(ROWS, COLS))
            x_full = np.random.randint(0, MAX_VAL, size=ROWS)
            
            with open("python_to_verilog_ideal.txt", "w") as f_out_ideal, open("python_to_verilog_real.txt", "w") as f_out_real:
                for x_bit in range(slices):          
                    for w_bit in range(slices):      
                        slice_counter += 1
                        combined_shift = x_bit + w_bit

                        x_slice = extract_bit_slice(x_full, x_bit).astype(float) * V_READ
                        W_slice = extract_bit_slice(W_full, w_bit)

                        try:
                            # Build and run ideal
                            netlist_ideal = build_netlist(W_slice, x_slice, ROWS, COLS, LRS, 5.0e19, R_row_wire=ROW_WIRE_RESISTANCE, R_col_wire=COL_WIRE_RESISTANCE)
                            raw_output_ideal = run_ngspice(netlist_ideal)
                            currents_ideal   = parse_currents(raw_output_ideal, COLS)
                            
                            # Build and run real
                            netlist_real = build_netlist(W_slice, x_slice, ROWS, COLS, LRS, 5.0e5, R_row_wire=ROW_WIRE_RESISTANCE, R_col_wire=COL_WIRE_RESISTANCE)
                            raw_output_real = run_ngspice(netlist_real)
                            currents_real   = parse_currents(raw_output_real, COLS)
                            
                            # Write to ideal file
                            for col_idx_ideal, current_ideal in enumerate(currents_ideal):
                                digit_val_ideal = (current_ideal * SCALING_FACTOR)
                                f_out_ideal.write(f"{col_idx_ideal} {digit_val_ideal:.6f} {combined_shift}\n")
                                
                            # Write to real file
                            for col_idx_real, current_real in enumerate(currents_real):
                                digit_val_real = (current_real * SCALING_FACTOR)
                                f_out_real.write(f"{col_idx_real} {digit_val_real:.6f} {combined_shift}\n")
                                
                        except Exception as e:
                            print(f"  Error in slice (x_bit={x_bit}, w_bit={w_bit}): {e}")

            # -------------------------------------------------------------------------
            # Hand off to Verilog (Shift-and-Add)
            # -------------------------------------------------------------------------
            try:
                subprocess.run(r"C:\iverilog\bin\iverilog -g2012 -o sim_ideal.vvp shift_add_tb_ideal.v", shell=True, check=True, stdout=subprocess.DEVNULL)
                subprocess.run(r"C:\iverilog\bin\vvp sim_ideal.vvp", shell=True, check=True, stdout=subprocess.DEVNULL)

                subprocess.run(r"C:\iverilog\bin\iverilog -g2012 -o sim_real.vvp shift_add_tb_real.v", shell=True, check=True, stdout=subprocess.DEVNULL)
                subprocess.run(r"C:\iverilog\bin\vvp sim_real.vvp", shell=True, check=True, stdout=subprocess.DEVNULL)

            except subprocess.CalledProcessError:
                print("Error: Verilog compilation or execution failed.")

            # -------------------------------------------------------------------------
            # Read hardware results back and accumulate error 
            # -------------------------------------------------------------------------
            file_ideal = "verilog_to_python_ideal.txt"
            file_real  = "verilog_to_python_real.txt"

            if os.path.exists(file_ideal) and os.path.exists(file_real):
                
                # These variables represent the INNER sum (Layer 2)
                cycle_total_error = 0.0
                valid_columns_in_cycle = 0
                
                with open(file_ideal, "r") as f_in_ideal, open(file_real, "r") as f_in_real:
                    for line1, line2 in zip(f_in_ideal, f_in_real):
                        if line1.strip() and line2.strip():
                            parts_ideal = line1.split()
                            parts_real  = line2.split()
                            
                            a_j = float(parts_ideal[1])
                            a_j_tilde = float(parts_real[1])
                            
                            if a_j != 0:
                                # LAYER 1: The individual element error
                                column_error = abs(a_j - a_j_tilde) / abs(a_j)
                                cycle_total_error += column_error
                                valid_columns_in_cycle += 1
                
                # LAYER 2: Calculate the average error for THIS single cycle (1/N * Sum)
                if valid_columns_in_cycle > 0:
                    cycle_average_error = cycle_total_error / valid_columns_in_cycle
                    
                    # LAYER 3: Add this 1 cycle's error to our 100-run master tracker
                    cumulative_error_100_runs += cycle_average_error
                    total_cycles_evaluated += 1

        # -------------------------------------------------------------------------
        # Final Verification Printout & Data Collection
        # -------------------------------------------------------------------------
        if total_cycles_evaluated > 0:
            
            # LAYER 3 CONTINUED: Divide by 100 (1/100 * Sum)
            final_average_error = cumulative_error_100_runs / total_cycles_evaluated
            error_percentage = final_average_error * 100
            
            print(f"\n=== RESULTS FOR SIZE {ROWS}x{COLS} (Averaged over {total_cycles_evaluated} cycles) ===")
            print(f"  Average Relative Error:        {final_average_error:.6f}")
            print(f"  Accuracy Percentage:           {100 - error_percentage:.4f}%")
            print("========================================================\n")
            
            # --- SAVE DATA FOR GRAPHING ---
            plot_sizes.append(size)
            plot_errors.append(final_average_error)
            
        else:
            print(f"Error: No valid data found to calculate error for size {size}.")


    # -------------------------------------------------------------------------
    # Generate the Final Graph
    # -------------------------------------------------------------------------
    if plot_sizes and plot_errors:
        print("Generating Error Analysis Graph...")
        
        plt.figure(figsize=(10, 6))
        
        # Plot the line with markers
        plt.plot(plot_sizes, plot_errors, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)
        
        # Formatting the axes and title
        plt.title('Crossbar Matrix Multiplication: Average Relative Error vs Matrix Size', fontsize=14, fontweight='bold')
        plt.xlabel('Matrix Size (N x N)', fontsize=12)
        plt.ylabel('Average Relative Error', fontsize=12)
        
        # Use a logarithmic scale for X because the matrix sizes grow exponentially
        plt.xscale('log', base=2)
        
        # Force the X-axis to label exact matrix sizes instead of generic exponents
        plt.xticks(plot_sizes, [f"{s}" for s in plot_sizes])
        
        # Add a grid for readability
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        
        plt.tight_layout()
        plt.show()
