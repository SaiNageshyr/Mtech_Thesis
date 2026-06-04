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
    ROWS     = int(input("Enter number of ROWS: "))
    COLS     = int(input("Enter number of COLUMNS: "))
    slices = 4#int(input("Enter number of slices: "))

    ROW_WIRE_RESISTANCE = 0.1
    COL_WIRE_RESISTANCE = 0.1
    HRS            = 5.0e19
    LRS            = 5.0e3
    SCALING_FACTOR = 50000
    V_READ         = 0.1

    # -------------------------------------------------------------------------
    # Generate ONE random integer W and x (values 0 .. 2^slices - 1)
    # -------------------------------------------------------------------------
    MAX_VAL = (1 << slices)   # 2^slices

    W_full =np.random.randint(0, MAX_VAL, size=(ROWS, COLS))
    x_full =np.random.randint(0, MAX_VAL, size=ROWS)

    print("\n" + "#" * 60)
    print("  ORIGINAL (full integer) inputs")
    print("#" * 60)
    print(f"\n  W =\n{W_full}")
    print(f"\n  x = {x_full}\n")



    # -------------------------------------------------------------------------
    # Nested slice loop:  x_bit in [0..slices-1]
    #                     w_bit in [0..slices-1]
    # Total NgSPICE runs = slices * slices
    # -------------------------------------------------------------------------
    total_slices = slices * slices
    slice_counter = 0

    with open("python_to_verilog.txt", "w") as f_out:

        for x_bit in range(slices):          # LSB → MSB of x
            for w_bit in range(slices):      # LSB → MSB of W

                slice_counter += 1
                combined_shift = x_bit + w_bit

                print(f"\n{'#'*60}")
                print(f"###  SLICE {slice_counter}/{total_slices}  "
                      f"|  x_bit={x_bit}  w_bit={w_bit}  "
                      f"|  combined_shift={combined_shift}  ###")
                print(f"{'#'*60}")

                # ── Extract single-bit slices ──────────────────────────────
                x_slice = extract_bit_slice(x_full, x_bit).astype(float) * V_READ
                W_slice = extract_bit_slice(W_full, w_bit)

                # ── Run NgSPICE ────────────────────────────────────────────
                try:
                    netlist = build_netlist(
                        W_slice, x_slice, ROWS, COLS, LRS, HRS,
                        R_row_wire=ROW_WIRE_RESISTANCE,
                        R_col_wire=COL_WIRE_RESISTANCE
                    )
                    print("Generated Netlist. Running NgSPICE...")

                    raw_output = run_ngspice(netlist)
                    currents   = parse_currents(raw_output, COLS)

                    print_results(W_slice, x_slice, currents, x_bit, w_bit)
                    python_cal(W_slice, x_slice, ROWS, COLS, HRS, LRS)
                    # ── Digitize and write to file ─────────────────────────
                    for col_idx, current in enumerate(currents):
                        digit_val = (current * SCALING_FACTOR)
                        # Format: col_index  digit_val  combined_shift
                        f_out.write(f"{col_idx} {digit_val} {combined_shift}\n")

                except Exception as e:
                    print(f"  Error in slice (x_bit={x_bit}, w_bit={w_bit}): {e}")

    # -------------------------------------------------------------------------
    # Hand off to Verilog (Shift-and-Add)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("=== Handing Data to Verilog for Shift-and-Add ===")
    print("=" * 60)

    try:
        print("Compiling Verilog...")
        subprocess.run(
            r"C:\iverilog\bin\iverilog -g2012 -o sim.vvp shift_add_tb.v",
            shell=True, check=True
        )
        print("Running Verilog simulation...")
        subprocess.run(r"C:\iverilog\bin\vvp sim.vvp", shell=True, check=True)

    except subprocess.CalledProcessError:
        print("Error: Verilog compilation or execution failed.")

    # -------------------------------------------------------------------------
    # Run ideal Python MVM for verification at the end
    # -------------------------------------------------------------------------
    ideal_result = python_mvm(W_full, x_full, COLS)

    # -------------------------------------------------------------------------
    # Read hardware results back and compare
    # -------------------------------------------------------------------------
    if os.path.exists("verilog_to_python.txt"):
        print("\n" + "=" * 60)
        print("=== FINAL COMPARISON: Verilog vs Ideal Python ===")
        print("=" * 60)
        print(f"\n  {'Column':<10} {'Verilog':>12} {'Ideal Python':>14}")
        print("  " + "-" * 48)

        with open("verilog_to_python.txt", "r") as f_in:
            for line in f_in:
                if line.strip():
                    parts      = line.split()
                    col_idx    = int(parts[0])
                    verilog_val = float(parts[1])
                    ideal_val  = int(ideal_result[col_idx])
                    print(f"  col_{col_idx+1:<6} {verilog_val:>12} {ideal_val:>14}")

        print("=" * 60 + "\n")
    else:
        print("Error: Verilog did not produce the output file.")
