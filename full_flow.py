import os
import numpy as np


folder_path = r"D:\CouchEd_projects\CouchEd\ngspice-42_64\Spice64\bin"
file_name = "crossbar_array_1_slice_line_res.cir"
full_path = os.path.join(folder_path, file_name)

def build_netlist(
    W_resistances: np.ndarray,
    x: np.ndarray,
    rows: int,
    cols: int,
    LRS: float,
    HRS: float,
    R_row_wire: float = 1.0,   # Line resistance along each row segment (Ohms)
    R_col_wire: float = 1.0    # Line resistance along each column segment (Ohms)
) -> str:

    lines = []
    lines.append("* Resistor Crossbar MVM — NgSPICE Simulation (with Line Resistances)")
    lines.append(f"* Rows={rows}, Cols={cols}")
    lines.append(f"* R_row_wire={R_row_wire} Ohm/segment, R_col_wire={R_col_wire} Ohm/segment")
    lines.append("")

    # ── Input voltage sources ────────────────────────────────────────────────
    # Each source drives the first node of its row wire: row_{i}_seg_0
    lines.append("* ── Input voltage sources ──────────────────────────────")
    for i in range(rows):
        lines.append(f"V_in_{i+1} row_{i+1}_seg_0 0 DC {float(x[i]):.6g}")
    lines.append("")

    # ── Row wire segment resistors ───────────────────────────────────────────
    # For row i: row_{i}_seg_0 → row_{i}_seg_1 → ... → row_{i}_seg_{cols-1}
    # Segment k connects seg_k to seg_{k+1}  (cols-1 segments total)
    lines.append("* ── Row wire segment resistors ─────────────────────────")
    for i in range(rows):
        for k in range(cols - 1):
            node_a = f"row_{i+1}_seg_{k}"
            node_b = f"row_{i+1}_seg_{k+1}"
            lines.append(f"R_row_{i+1}_seg_{k+1}  {node_a}  {node_b}  {R_row_wire:.6g}")
        lines.append("")

    # ── Column wire segment resistors ────────────────────────────────────────
    # For col j: col_{j}_seg_0 → col_{j}_seg_1 → ... → col_{j}_seg_{rows-1}
    # Segment k connects seg_k to seg_{k+1}  (rows-1 segments total)
    lines.append("* ── Column wire segment resistors ──────────────────────")
    for j in range(cols):
        for k in range(rows - 1):
            node_a = f"col_{j+1}_seg_{k}"
            node_b = f"col_{j+1}_seg_{k+1}"
            lines.append(f"R_col_{j+1}_seg_{k+1}  {node_a}  {node_b}  {R_col_wire:.6g}")
        lines.append("")

    # ── Crossbar resistors ───────────────────────────────────────────────────
    # Device at (row i, col j) connects:
    #   row node  : row_{i+1}_seg_{j}   (j-th segment node along row i)
    #   col node  : col_{j+1}_seg_{i}   (i-th segment node along col j)
    lines.append("* ── Crossbar Resistors ──────────────────────────────────")
    for i in range(rows):
        lines.append(f"*          row{i+1}")
        for j in range(cols):
            if W_resistances[i][j] == 0:
                lines.append(f"R_r{i+1}c{j+1} row_{i+1}_seg_{j}  col_{j+1}_seg_{i} {HRS:.6g}")
            else:
                lines.append(f"R_r{i+1}c{j+1}  row_{i+1}_seg_{j}  col_{j+1}_seg_{i}  {LRS:.6g}")
        lines.append("")

    # ── Ammeters ─────────────────────────────────────────────────────────────
    # Connected at the last segment node of each column wire
    lines.append("* ── Ammeters (0-V sources for current sensing) ─────────")
    for j in range(cols):
        last_col_node = f"col_{j+1}_seg_{rows-1}"
        lines.append(f"V_ammeter_{j+1}  {last_col_node}  0  DC 0")
    lines.append("")

    # ── Analysis ─────────────────────────────────────────────────────────────
    lines.append("* ── Analysis ────────────────────────────────────────────")
    lines.append(".op")
    lines.append("")

    ammeter_list = " ".join(f"I(V_ammeter_{j+1})" for j in range(cols))
    lines.append("* ── Output ──────────────────────────────────────────────")
    lines.append(".control")
    lines.append("run")
    lines.append(f"print {ammeter_list}")
    lines.append(".endc")
    lines.append("")
    lines.append(".end")

    return "\n".join(lines)


if __name__ == "__main__":
    ROWS = 2
    COLS = 2
    
    LRS = 5.0e+3
    HRS = 5.0e+9

    W = np.random.randint(2, size=(ROWS, COLS))

    x = np.array([0, 1])

    # ── Set your wire resistance values here ─────────────────────────────────
    # These represent resistance per segment between adjacent crosspoints.
    # For a physical wire: R_segment = rho * L / A
    # Typical values for metal interconnects range from milli-Ohms to a few Ohms.
    ROW_WIRE_RESISTANCE = 1.0   # Ohms per segment along a row
    COL_WIRE_RESISTANCE = 1.0   # Ohms per segment along a column

    netlist = build_netlist(W, x, ROWS, COLS, LRS, HRS,
                            R_row_wire=ROW_WIRE_RESISTANCE,
                            R_col_wire=COL_WIRE_RESISTANCE)

    print("=" * 60)
    print("Generated Netlist:")
    print("=" * 60)

    try:
        with open(full_path, "w", encoding="utf-8") as file:
            file.write(netlist)
        print(f"\nSuccess! Circuit file saved as: {full_path}")
    except Exception as e:
        print(f"\nFailed to save file: {e}")
