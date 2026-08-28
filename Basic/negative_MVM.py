'''import os
import re
import subprocess
import tempfile
import numpy as np

# -----------------------------------------------------------------------------
#  NgSPICE & Iverilog settings (LINUX COMPATIBLE)
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
    lines.append("\n.end")
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
    clean_out = ngspice_output.replace("Using SPARSE 1.3 as Direct Linear Solver", "")
    clean_out = re.sub(r'\s+', '', clean_out)
    for j in range(cols):
        pattern = r"i\(v_ammeter_{}\)=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)".format(j+1)
        match = re.search(pattern, clean_out, re.IGNORECASE)
        if match:
            currents[j] = float(match.group(1))
    return currents

# -----------------------------------------------------------------------------
#  Math Helpers
# -----------------------------------------------------------------------------
def extract_bit_slice(matrix: np.ndarray, bit_pos: int) -> np.ndarray:
    return ((matrix >> bit_pos) & 1).astype(int)

def adc_quantise(val: float, step: float) -> float:
    return float(np.round(val / step) * step)

# -----------------------------------------------------------------------------
#  Main Loop
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    
    ROWS     = int(input("Enter number of ROWS: "))
    COLS     = int(input("Enter number of COLUMNS: "))
    slices   = int(input("Enter number of slices: "))
    adc_mode = int(input("Enter ADC mode (1: No ADC, 2: With ADC, 3: Both): "))

    if adc_mode == 1:
        adc_configs = [False]  
    elif adc_mode == 2:
        adc_configs = [True]   
    elif adc_mode == 3:
        adc_configs = [True,False]  
    else:
        print("Invalid choice, defaulting to Both (3).")
        adc_configs = [False, True]

    ROW_WIRE_RESISTANCE = 0.1
    COL_WIRE_RESISTANCE = 0.1
    HRS            = 5.0e19
    LRS            = 5.0e3
    V_READ         = 0.1
    
    # ADC Scaling Factors mapping to the Physical Conductance domain
    SCALING_FACTOR = (1 / V_READ) 
    SF = LRS
    ADC_STEP       = 1 / LRS 
    
    MAX_VAL = (1 << slices) 
    
    # 1. Generate SIGNED random matrices
    W_full =np.array([[4, 4, 6],[11, 5, 12],[-7, 8, 13]]) #np.random.randint(-MAX_VAL + 1, MAX_VAL, size=(ROWS, COLS))
    x_full =np.array([-1, -7, 7]) #np.random.randint(-MAX_VAL + 1, MAX_VAL, size=ROWS)

    print("\n" + "#" * 60)
    print("  SIGNED (full integer) inputs")
    print("#" * 60)
    print(f"\n  W =\n{W_full}")
    print(f"\n  x = {x_full}\n")

    # 2. Split W into Positive and Negative arrays
    W_mag = np.abs(W_full)
    W_pos = np.where(W_full > 0, W_mag, 0)
    W_neg = np.where(W_full < 0, W_mag, 0)
    
    # Extract sign and magnitude of x
    x_mag = np.abs(x_full)
    x_sign = np.where(x_full < 0, -1.0, 1.0) 

    slice_counter = 0
    total_slices = slices * slices
    
    # Pre-calculate ideal result for later comparison
    ideal_result = W_full.T @ x_full

    # 3. SPICE Simulation Phase (Run all physical hardware steps first)
    cycle_slice_data = []

    for x_bit in range(slices):          
        for w_bit in range(slices):      
            slice_counter += 1
            combined_shift = x_bit + w_bit

            # Extract bit slices based purely on Magnitudes
            x_slice_mag = extract_bit_slice(x_mag, x_bit)
            W_pos_slice = extract_bit_slice(W_pos, w_bit)
            W_neg_slice = extract_bit_slice(W_neg, w_bit)
            
            # Apply the original sign of X to the input voltage! 
            x_volts = x_slice_mag.astype(float) * x_sign * V_READ

            try:
                # Run NgSPICE on Positive Crossbar
                nl_pos = build_netlist(W_pos_slice, x_volts, ROWS, COLS, LRS, HRS, ROW_WIRE_RESISTANCE, COL_WIRE_RESISTANCE)
                curr_pos = parse_currents(run_ngspice(nl_pos), COLS)

                # Run NgSPICE on Negative Crossbar
                nl_neg = build_netlist(W_neg_slice, x_volts, ROWS, COLS, LRS, HRS, ROW_WIRE_RESISTANCE, COL_WIRE_RESISTANCE)
                curr_neg = parse_currents(run_ngspice(nl_neg), COLS)

                # SUBTRACT: 
                net_current = curr_pos - curr_neg
                
                # Store raw physical current for the next step
                cycle_slice_data.append((combined_shift, net_current))

            except Exception as e:
                print(f"  Error in slice: {e}")
                
    # 4. ADC and Verilog Handoff Phase
    for use_adc in adc_configs:
        mode_str = "With ADC" if use_adc else "No ADC"
        print(f"\n=== Processing Mode: {mode_str} ===")
        
        with open("python_to_verilog.txt", "w") as f_out:
            for combined_shift, net_current in cycle_slice_data:
                # Convert Amperes to Siemens
                curr_scaled = net_current * SCALING_FACTOR
                
                for col_idx in range(COLS):
                    if use_adc:
                        val_to_write = adc_quantise(curr_scaled[col_idx], ADC_STEP)
                    else:
                        val_to_write = curr_scaled[col_idx]
                    
                    f_out.write(f"{col_idx} {val_to_write} {combined_shift}\n")

        # Execute Verilog
        try:
            subprocess.run(f"{IVERILOG_CMD} -g2012 -o sim.vvp shift_add_tb.v", shell=True, check=True, stdout=subprocess.DEVNULL)
            subprocess.run(f"{VVP_CMD} sim.vvp", shell=True, check=True, stdout=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            print("Error: Verilog execution failed.")
            continue

        # 5. Read Verilog Output and Scale Back to Digital
        hardware_accumulated = np.zeros(COLS)
        if os.path.exists("verilog_to_python.txt"):
            with open("verilog_to_python.txt", "r") as f_in:
                for line in f_in:
                    if line.strip():
                        parts = line.split()
                        col_idx = int(parts[0])
                        # Convert Siemens back to Dimensionless Digital Integer representation
                        verilog_val = float(parts[1]) * SF
                        hardware_accumulated[col_idx] = verilog_val
        else:
            raise RuntimeError("Verilog output file not found!")

        # Print Final Comparison
        print(f"=== FINAL COMPARISON: Signed MVM ({mode_str}) ===")
        print(f"  {'Column':<8} | {'Hardware Verilog':>16} | {'Ideal Python':>16}")
        print("  " + "-" * 50)

        for col_idx in range(COLS):
            verilog_val = hardware_accumulated[col_idx]
            ideal_val = int(ideal_result[col_idx])
            
            # Highlight exact matches
            print(f"  col_{col_idx+1:<4} | {verilog_val:>16.2f} | {ideal_val:>16}")
            
        print("=" * 60)'''

import os
import re
import subprocess
import tempfile
import numpy as np

# -----------------------------------------------------------------------------
#  NgSPICE & Iverilog settings (LINUX COMPATIBLE)
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
    lines.append("\n.end")
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
    clean_out = ngspice_output.replace("Using SPARSE 1.3 as Direct Linear Solver", "")
    clean_out = re.sub(r'\s+', '', clean_out)
    for j in range(cols):
        pattern = r"i\(v_ammeter_{}\)=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)".format(j+1)
        match = re.search(pattern, clean_out, re.IGNORECASE)
        if match:
            currents[j] = float(match.group(1))
    return currents

# -----------------------------------------------------------------------------
#  Math Helpers
# -----------------------------------------------------------------------------
def extract_bit_slice(matrix: np.ndarray, bit_pos: int) -> np.ndarray:
    return ((matrix >> bit_pos) & 1).astype(int)

def adc_quantise(val: float, step: float) -> float:
    # 1. Inject a physical +0.5 LSB analog voltage bias
    analog_offset = step / 2.0
    # 2. The comparator strictly truncates (floors) the biased signal
    clean_ratio = (val + analog_offset) / step
    return float(np.floor(clean_ratio) * step)

def to_2s_complement(matrix: np.ndarray, bits: int) -> np.ndarray:
    return np.where(matrix < 0, (1 << bits) + matrix, matrix)

# -----------------------------------------------------------------------------
#  Main Loop
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    
    ROWS     = int(input("Enter number of ROWS: "))
    COLS     = int(input("Enter number of COLUMNS: "))
    slices   = int(input("Enter number of slices (e.g., 4 or 8): "))
    adc_mode = int(input("Enter ADC mode (1: No ADC, 2: With ADC, 3: Both): "))

    if adc_mode == 1:
        adc_configs = [False]  
    elif adc_mode == 2:
        adc_configs = [True]   
    elif adc_mode == 3:
        adc_configs = [False, True]  
    else:
        print("Invalid choice, defaulting to Both (3).")
        adc_configs = [False, True] 

    ROW_WIRE_RESISTANCE = 0.1
    COL_WIRE_RESISTANCE = 0.1
    HRS            = 5.0e5
    LRS            = 5.0e3
    V_READ         = 0.1
    
    SCALING_FACTOR = (1 / V_READ) 
    SF = LRS
    ADC_STEP       = 1 / LRS 
    
    # -------------------------------------------------------------------------
    # 2's Complement Bound Logic
    # -------------------------------------------------------------------------
    MIN_VAL = -(1 << (slices - 1))
    MAX_VAL = (1 << (slices - 1)) - 1
    
    W_full = np.random.randint(MIN_VAL, MAX_VAL + 1, size=(ROWS, COLS))
    x_full = np.random.randint(MIN_VAL, MAX_VAL + 1, size=ROWS)

    print("\n" + "#" * 60)
    print(f"  SIGNED RANDOM inputs ({slices}-bit limit: {MIN_VAL} to {MAX_VAL})")
    print("#" * 60)
    print(f"\n  W =\n{W_full}")
    print(f"\n  x = {x_full}\n")

    # Pre-calculate ideal math for comparison
    ideal_result = W_full.T @ x_full

    # Convert raw negative numbers to pure 2's Complement integers
    W_2c = to_2s_complement(W_full, slices)
    x_2c = to_2s_complement(x_full, slices)

    slice_counter = 0
    cycle_slice_data = []

    # 3. SPICE Simulation Phase (ANALOG)
    for x_bit in range(slices):          
        for w_bit in range(slices):      
            slice_counter += 1
            combined_shift = x_bit + w_bit

            # Extract bits directly from the 2's complement integers
            x_slice = extract_bit_slice(x_2c, x_bit)
            W_slice = extract_bit_slice(W_2c, w_bit)
            
            # Map '1' to V_READ and '0' to GND
            x_volts = x_slice.astype(float) * V_READ

            # -----------------------------------------------------------------
            # BAUGH-WOOLEY MULTIPLIER LOGIC (Control Signals Only)
            # -----------------------------------------------------------------
            is_x_msb = (x_bit == slices - 1)
            is_w_msb = (w_bit == slices - 1)

            if is_x_msb != is_w_msb:
                slice_sign = -1.0 # Tells Digital ALU to Subtract later
            else:
                slice_sign = 1.0  # Tells Digital ALU to Add later

            try:
                # Run ONE crossbar. It naturally outputs purely POSITIVE currents!
                nl = build_netlist(W_slice, x_volts, ROWS, COLS, LRS, HRS, ROW_WIRE_RESISTANCE, COL_WIRE_RESISTANCE)
                
                # Raw, positive analog currents straight from the crossbar ammeters
                raw_currents = parse_currents(run_ngspice(nl), COLS)
                
                # We save both the raw positive current AND the sign control flag
                cycle_slice_data.append((combined_shift, raw_currents, slice_sign))

            except Exception as e:
                print(f"  Error in slice: {e}")
                
    # 4. ADC and Verilog Handoff Phase (DIGITAL)
    for use_adc in adc_configs:
        mode_str = "With ADC" if use_adc else "No ADC"
        print(f"\n=== Processing Mode: {mode_str} ===")
        
        with open("python_to_verilog.txt", "w") as f_out:
            # Unpack the stored variables: shift, positive currents, and ALU sign flag
            for combined_shift, raw_currents, slice_sign in cycle_slice_data:
                
                # Convert raw positive Amperes to positive Siemens
                curr_scaled = raw_currents * SCALING_FACTOR
                
                for col_idx in range(COLS):
                    if use_adc:
                        # 1. ADC Quantizes the POSITIVE analog magnitude
                        quantized_mag = adc_quantise(curr_scaled[col_idx], ADC_STEP)
                    else:
                        quantized_mag = curr_scaled[col_idx]
                    
                    # 2. The Digital ALU applies the Baugh-Wooley Sign logic (Add or Subtract)
                    val_to_write = quantized_mag * slice_sign
                    
                    f_out.write(f"{col_idx} {val_to_write} {combined_shift}\n")

        # Execute Verilog
        try:
            subprocess.run(f"{IVERILOG_CMD} -g2012 -o sim.vvp shift_add_tb.v", shell=True, check=True, stdout=subprocess.DEVNULL)
            subprocess.run(f"{VVP_CMD} sim.vvp", shell=True, check=True, stdout=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            print("Error: Verilog execution failed.")
            continue

        # 5. Read Verilog Output and Scale Back to Digital
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
            raise RuntimeError("Verilog output file not found!")

        # Print Final Comparison
        print(f"=== FINAL COMPARISON: 2's Comp MVM ({mode_str}) ===")
        print(f"  {'Column':<8} | {'Hardware Verilog':>16} | {'Ideal Python':>16}")
        print("  " + "-" * 50)

        for col_idx in range(COLS):
            verilog_val = hardware_accumulated[col_idx]
            ideal_val = int(ideal_result[col_idx])
            
            # Highlight exact matches
            match_str = "✓" if round(verilog_val) == ideal_val else ""
            print(f"  col_{col_idx+1:<4} | {verilog_val:>16.2f} | {ideal_val:>16} {match_str}")
            
        print("=" * 60)
