import os
import numpy as np

folder_path = r"D:\CouchEd_projects\CouchEd\ngspice-42_64\Spice64\bin"
file_name = "crossbar_array_n_slices_line_res.cir"
full_path = os.path.join(folder_path, file_name)

def build_netlist(
    W_resistance: np.ndarray,
    x: np.ndarray,
    slices: int,
    rows: int,
    cols: int,
    R_row_wire: float = 1.0,   
    R_col_wire: float = 1.0    
) -> str:


    lines = []
    lines.append("* Resistor Crossbar MVM — NgSPICE Simulation (with Line Resistances)")
    lines.append(f"* Slices={slices}, Rows={rows}, Cols={cols}")
    lines.append(f"* R_row_wire={R_row_wire} Ohm/segment, R_col_wire={R_col_wire} Ohm/segment")
    lines.append("")

    # ── Input voltage sources ────────────────────────────────────────────────
    lines.append("* ── Input voltage sources ──────────────────────────────")
    for k in range(slices):
        lines.append(f"*          slice{k+1}")
        for i in range(rows):
            lines.append(f"V_in_{k+1}_{i+1}  row_s{k+1}_r{i+1}_seg_0  0  DC {float(x[i]):.6g}")
        lines.append("")

    # ── Row wire segment resistors ───────────────────────────────────────────
    lines.append("* ── Row wire segment resistors ─────────────────────────")
    for k in range(slices):
        lines.append(f"*          slice{k+1}")
        for i in range(rows):
            lines.append(f"*          row{i+1}")
            for seg in range(cols - 1):
                lines.append(f"R_row_s{k+1}_r{i+1}_seg{seg+1}  row_s{k+1}_r{i+1}_seg_{seg}  row_s{k+1}_r{i+1}_seg_{seg+1}  {R_row_wire:.6g}")
            lines.append("")
        lines.append("")

    # ── Column wire segment resistors ────────────────────────────────────────
    lines.append("* ── Column wire segment resistors ──────────────────────")
    for k in range(slices):
        lines.append(f"*          slice{k+1}")
        for j in range(cols):
            lines.append(f"*          col{j+1}")
            for seg in range(rows - 1):
                lines.append(f"R_col_s{k+1}_c{j+1}_seg{seg+1}  col_s{k+1}_c{j+1}_seg_{seg}  col_s{k+1}_c{j+1}_seg_{seg+1}  {R_col_wire:.6g}")
            lines.append("")
        lines.append("")

    # ── Crossbar resistors ───────────────────────────────────────────────────
    lines.append("* ── Crossbar Resistors ──────────────────────────────────")
    for k in range(slices):
        lines.append(f"*          slice{k+1}")
        for i in range(rows):
            lines.append(f"*          row{i+1}")
            for j in range(cols):
                if W_resistance[i][j] ==0:
                    lines.append(f"R_s{k+1}r{i+1}c{j+1}  row_s{k+1}_r{i+1}_seg_{j}  col_s{k+1}_c{j+1}_seg_{i}  {5.0e+9}")
                else:
                    lines.append(f"R_s{k+1}r{i+1}c{j+1}  row_s{k+1}_r{i+1}_seg_{j}  col_s{k+1}_c{j+1}_seg_{i}  {5.0e+3}")
            lines.append("")
        lines.append("")

    # ── Ammeters ─────────────────────────────────────────────────────────────
    lines.append("* ── Ammeters (0-V sources for current sensing) ─────────")
    for k in range(slices):
        lines.append(f"*          slice{k+1}")
        for j in range(cols):
            last_col_node = f"col_s{k+1}_c{j+1}_seg_{rows-1}"
            lines.append(f"V_ammeter_{k+1}_{j+1}  {last_col_node}  0  DC 0")
        lines.append("")

    # ── Analysis ─────────────────────────────────────────────────────────────
    lines.append("* ── Analysis ────────────────────────────────────────────")
    lines.append(".op")
    lines.append("")

    ammeter_list = []
    for k in range(slices):
        for j in range(cols):
            ammeter_list.append(f"I(V_ammeter_{k+1}_{j+1})")

    #print("Ammeter list:", ammeter_list)

    lines.append("* ── Output ──────────────────────────────────────────────")
    lines.append(".control")
    lines.append("run")
    lines.append(f"print {' '.join(ammeter_list)}")
    lines.append(".endc")
    lines.append("")
    lines.append(".end")

    return "\n".join(lines)


if __name__ == "__main__":
    ROWS = 2
    COLS = 2
    no_slices = 2

    # Conductance matrix (G) in Siemens — shared across all slices
    W = np.array([
        [0, 1],
        [1, 0]
    ])
    x = np.array([0, 1])
    # ── Set your wire resistance values here ─────────────────────────────────
    ROW_WIRE_RESISTANCE = 1.0   # Ohms per segment along a row
    COL_WIRE_RESISTANCE = 1.0   # Ohms per segment along a column

    netlist = build_netlist(W, x, no_slices, ROWS, COLS,R_row_wire=ROW_WIRE_RESISTANCE,R_col_wire=COL_WIRE_RESISTANCE)
    print("=" * 60)
    print("Generated Netlist:")
    try:
        with open(full_path, "w", encoding="utf-8") as file:
            file.write(netlist)
        print(f"\nSuccess! Circuit file saved as: {full_path}")
    except Exception as e:
        print(f"\nFailed to save file: {e}")
