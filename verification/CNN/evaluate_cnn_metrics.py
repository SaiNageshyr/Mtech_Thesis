import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt

def get_pred(name, text):
    m = re.search(name + r"\s*:\s*(\d+)", text, re.IGNORECASE)
    return int(m.group(1)) if m else None

if __name__ == "__main__":
    bits_list = [4, 8, 16, 32]
    hrs_strings = ["5.0e4", "5.0e5", "5.0e19"]
    csv_file = "cnn_misclassification_data.csv"
    rows = []

    print("Starting CNN batch evaluation...")

    for hrs_str in hrs_strings:
        hrs_val = float(hrs_str)
        for bits in bits_list:
            log_dir = f"logs_cnn_hrs_{hrs_str}_{bits}bit"
            log_files = glob.glob(os.path.join(log_dir, "log_img_*.txt"))
            if not log_files:
                continue

            c = dict(total=0, flt_hw=0, flt_ideal=0, ideal_hw=0, hw_true=0)
            skipped = 0

            for log_file in log_files:
                try:
                    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                except Exception as e:
                    print(f"  WARNING: could not read {log_file}: {e}")
                    skipped += 1
                    continue

                true = get_pred("TRUE LABEL", content)
                flt = get_pred("FLOAT NN MODEL PREDICTION", content)
                ideal = get_pred("IDEAL PYTHON PREDICTION", content)
                hw = get_pred("HARDWARE VERILOG PREDICTION", content)

                if None in (true, flt, ideal, hw):
                    skipped += 1
                    continue

                c["total"] += 1
                c["flt_hw"] += int(flt != hw)          
                c["flt_ideal"] += int(flt != ideal)    
                c["ideal_hw"] += int(ideal != hw)      
                c["hw_true"] += int(hw != true)        

            if c["total"] > 0:
                msg = (f"Processed {log_dir:<28} | Total: {c['total']} | float!=hw: {c['flt_hw']:>3} | "
                       f"float!=ideal: {c['flt_ideal']:>3} | ideal!=hw: {c['ideal_hw']:>3} | hw!=true: {c['hw_true']:>3}")
                if skipped:
                    msg += f" | SKIPPED logs: {skipped}"
                print(msg)
                rows.append({
                    "Bits": bits, "HRS": hrs_val, "Total": c["total"],
                    "Misclassifications": c["flt_hw"],   
                    "Float_vs_Ideal": c["flt_ideal"],
                    "Ideal_vs_HW": c["ideal_hw"],
                    "HW_vs_True": c["hw_true"],
                })

    if not rows:
        print("\nNo valid data found to process or plot.")
    else:
        df = pd.DataFrame(rows)
        df.to_csv(csv_file, index=False)

        def plot_metric(ax, col, title, ylabel):
            for hrs_val, group in df.groupby("HRS"):
                group = group.sort_values(by="Bits")
                ax.plot(group["Bits"], group[col], marker='o', linewidth=2.5, markersize=8,
                        label=rf'HRS = {hrs_val:.1e} $\Omega$')
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.set_xlabel("Quantization Bit Width", fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_xticks(sorted(df["Bits"].unique()))
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(title="HRS Mean Variability", fontsize=9)

        fig, ax = plt.subplots(figsize=(9, 6))
        plot_metric(ax, "Misclassifications", "CNN Hardware Misclassifications vs. Bit Precision",
                    "Total Misclassifications (Float NN vs Hardware)")
        fig.tight_layout()
        fig.savefig("cnn_misclassifications_vs_bits_hrs_1000.png", dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 6))
        plot_metric(ax, "Ideal_vs_HW", "Pure Hardware Effect: Ideal Python vs Hardware",
                    "Mismatches (Ideal vs Hardware)")
        fig.tight_layout()
        fig.savefig("cnn_ideal_vs_hw_vs_bits_hrs_1000.png", dpi=300)
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        plot_metric(axes[0], "Float_vs_Ideal", "Baseline: Float vs Ideal", "Mismatches")
        plot_metric(axes[1], "Ideal_vs_HW", "Hardware: Ideal vs HW", "Mismatches")
        plot_metric(axes[2], "HW_vs_True", "Accuracy: HW vs True Label", "Wrong predictions")
        fig.tight_layout()
        fig.savefig("cnn_metrics_breakdown_1000.png", dpi=200)
        plt.close(fig)

        print("\nUpdated CNN plots generated: cnn_misclassifications_vs_bits_hrs_1000.png, "
              "cnn_ideal_vs_hw_vs_bits_hrs_1000.png, cnn_metrics_breakdown_1000.png")
