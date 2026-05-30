import os
import re
import subprocess
import tempfile
import numpy as np

# -----------------------------------------------------------------------------
#  NgSPICE settings
# -----------------------------------------------------------------------------
NGSPICE_CMD   = r"D:\CouchEd_projects\CouchEd\ngspice-42_64\Spice64\bin\ngspice_con.exe"
NGSPICE_FLAGS = ["-b"]      
#IVERILOG_CMD = r"C:\iverilog\bin\iverilog.exe"
#VVP_CMD      = r"C:\iverilog\bin\vvp.exe"
# -----------------------------------------------------------------------------
#  Netlist builder
# -----------------------------------------------------------------------------
def build_netlist(W_resistances: np.ndarray, x: np.ndarray, rows: int, cols: int, LRS: float, HRS: float,
    R_row_wire: float = 1.0,   
    R_col_wire: float = 1.0    
) -> str:

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
        subprocess.run(cmd, capture_output=True, text=True, timeout=60)

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
#  Diagnostic Output parser
# -----------------------------------------------------------------------------
def parse_currents(ngspice_output: str, cols: int) -> np.ndarray:
    currents = np.zeros(cols)
    
    if "error" in ngspice_output.lower() or "fatal" in ngspice_output.lower():
        print("\n=== NGSPICE FATAL ERROR DETECTED ===")
        print(ngspice_output)
        print("====================================\n")
        raise RuntimeError("NgSPICE failed to simulate the circuit.")

    for j in range(cols):
        pattern = r"i\(v_ammeter_{}\)\s*[=\s]\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)".format(j+1)
        match = re.search(pattern, ngspice_output.lower())
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
def print_results(W, x, y):
    rows, cols = W.shape
    sep = "=" * 58

    print("\n" + sep)
    print(f"  Crossbar MVM Result  ({rows} rows x {cols} cols)")
    print(sep)

    print("\nInput voltage vector x:")
    for i, v in enumerate(x):
        print(f"  row_{i+1:<4}  {v:>14.6g}")

    print("\nMatrix W :")
    for i in range(rows):
        row_str = f"  row_{i+1}:  "
        for j in range(cols):
            row_str += f" {W[i][j]:>10}"
        print(row_str)

    print("\nNgSPICE output currents:")
    print(f"  {'Column':<8}  {'Current (A)':>16}  {'Current (uA)':>14}")
    print("  " + "-" * 42)
    for j, current in enumerate(y):
        print(f"  col_{j+1:<4}  {current:>16.6e}  {current * 1e6:>14.4f}")
    print("\n" + sep + "\n")

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
        
    return output_I
def calculate_ideal_full_mvm(W_dict, x_dict, slices_count, V_READ):
    rows, cols = W_dict[1].shape

    W_full = np.zeros((rows, cols), dtype=int)
    for j in range(slices_count):
        W_full += W_dict[j+1] * (1 << j)

    x_full = np.zeros(rows, dtype=int)
    for i in range(slices_count):
        x_unscaled = np.round(x_dict[i+1] / V_READ).astype(int)
        x_full += x_unscaled * (1 << i)

    ideal_result = W_full.T @ x_full

    print("\n" + "=" * 60)
    print("=== IDEAL PYTHON MVM RESULT (Reconstructed) ===")
    print("=" * 60)
    print(f"\nReconstructed W_full:\n{W_full}")
    print(f"\nReconstructed x_full:\n{x_full}")
    
    print("\nIdeal Accumulated Digital Sums (W^T * x):")
    for col_idx, val in enumerate(ideal_result):
        print(f"  Column {col_idx+1:<4} -> {val}")
    print("=" * 60 + "\n")

    return ideal_result    
# -----------------------------------------------------------------------------
#  Main Loop with Verilog Integration
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    
    ROWS   = int(input("Enter number of ROWS: "))
    COLS   = int(input("Enter number of COLUMNS: "))
    slices = int(input("Enter number of slices: "))
    ROW_WIRE_RESISTANCE = 0.1
    COL_WIRE_RESISTANCE = 0.1
    HRS = 5.0e19 #float(input("Enter the High Resistance State value"))
    LRS = 5.0e3#float(input("Enter the Low Resistance State value"))
    
    W_ = {}
    x_ = {}
    
    SCALING_FACTOR = 50000 
    V_READ = 0.1
    
    # Open the text file to write the digitized currents for Verilog
    with open("python_to_verilog.txt", "w") as f_out:
        
        for k in range(slices):  
            W_[k+1] = np.random.randint(2, size=(ROWS, COLS))
            # Multiplying by V_READ ensures the input vector matches your 0.1V logic
            x_[k+1] = np.random.randint(2, size=ROWS) * V_READ
            
        for i in range(slices):
            for j in range(slices):
                try:
                    netlist = build_netlist(W_[j+1], x_[i+1], ROWS, COLS, LRS, HRS, R_row_wire=ROW_WIRE_RESISTANCE, R_col_wire=COL_WIRE_RESISTANCE)
                    print("Generated Netlist successfully. Running NgSPICE simulation...")
                    
                    raw_output = run_ngspice(netlist)
                    currents   = parse_currents(raw_output, COLS)
                    
                    print_results(W_[j+1], x_[i+1], currents)
                    python_cal(W_[j+1], x_[i+1], ROWS, COLS, HRS, LRS)
                    
                    # --- DIGITIZE AND WRITE TO FILE ---
                    shift_amount = i+j
                    
                    for col_idx, current in enumerate(currents):
                        digit_val = round(current * SCALING_FACTOR)
                        # Format: column_index  digitized_current  shift_amount
                        f_out.write(f"{col_idx} {digit_val} {shift_amount}\n")
                        
                except Exception as e:
                    print(f"An error occurred: {e}")

    # -------------------------------------------------------------------------
    # Hardware Co-Simulation (Verilog Execution)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("=== Handing Data to Verilog for Shift-and-Add ===")
    print("=" * 60)

    try:
        print("Compiling Verilog...")
        
        subprocess.run(r"C:\iverilog\bin\iverilog -g2012 -o sim.vvp shift_add_tb.v", shell=True, check=True)
        
        print("Running Verilog simulation...")
        subprocess.run(r"vvp sim.vvp", shell=True, check=True)
        
    except subprocess.CalledProcessError:
        print("Error: Verilog compilation or execution failed.")
    ideal_results = calculate_ideal_full_mvm(W_, x_, slices, V_READ)
    # -------------------------------------------------------------------------
    # Read Hardware Results Back
    # -------------------------------------------------------------------------
    if os.path.exists("verilog_to_python.txt"):
        print("\n=== FINAL HARDWARE RESULTS (From Verilog) ===")
        with open("verilog_to_python.txt", "r") as f_in:
            for line in f_in:
                if line.strip():
                    col_idx, total_sum = line.split()
                    print(f"  Column {int(col_idx)+1:<4} -> Accumulated Digital Sum: {total_sum}")
        print("=============================================\n")
    else:
        print("Error: Verilog did not produce the output file.")
