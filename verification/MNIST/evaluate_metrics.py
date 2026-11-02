import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    bits_list = [4, 8, 16, 32]
    #bits_list = [4, 8, 16]
    # Use exact strings to match the bash script's folder creation
    hrs_strings = ["5.0e4", "5.0e5", "5.0e19"] 
    #hrs_strings = ["5e19"] 
    csv_file = "misclassification_data.csv"
    batch_results = []

    print("Starting batch evaluation...")

    for hrs_str in hrs_strings:
        hrs_val = float(hrs_str) # Convert to float for CSV and plotting
        
        for bits in bits_list:
            # Construct the directory name using the exact string (e.g., logs_hrs_5.0e4_4bit)
            log_dir = f"logs_hrs_{hrs_str}_{bits}bit"
            log_files = glob.glob(os.path.join(log_dir, "log_img_*.txt"))

            if not log_files:
                print(f"  -> Skipping {log_dir} (No log files found)")
                continue

            match_float_nn = 0
            total_images = 0

            # Parse logs for this specific combination
            for log_file in log_files:
                try:
                    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()

                    float_pred_match = re.search(r"FLOAT NN MODEL PREDICTION\s*:\s*(\d+)", content, re.IGNORECASE)
                    hw_pred_match = re.search(r"HARDWARE VERILOG PREDICTION\s*:\s*(\d+)", content, re.IGNORECASE)

                    if not (float_pred_match and hw_pred_match):
                        continue

                    float_pred = int(float_pred_match.group(1))
                    hw_pred = int(hw_pred_match.group(1))

                    if hw_pred == float_pred:
                        match_float_nn += 1
                    total_images += 1

                except Exception as e:
                    print(f"DEBUG: Error reading {os.path.basename(log_file)}: {e}")

            if total_images > 0:
                misclassifications = total_images - match_float_nn
                print(f"Processed {log_dir:<22} | Total: {total_images} | Misclassifications: {misclassifications}")
                
                # Store the result using the float value
                batch_results.append({
                    "Bits": bits, 
                    "HRS": hrs_val, 
                    "Misclassifications": misclassifications
                })

    # Update the CSV file with the new batch data
    if batch_results:
        new_df = pd.DataFrame(batch_results)
        
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            # Overwrite previous data for the specific Bits/HRS combos being updated
            for _, row in new_df.iterrows():
                df = df[~((df['Bits'] == row['Bits']) & (df['HRS'] == row['HRS']))]
            df = pd.concat([df, new_df], ignore_index=True)
        else:
            df = new_df
            
        df.to_csv(csv_file, index=False)

        # Generate the multi-line plot using the aggregated data
        plt.figure(figsize=(9, 6))
        
        for hrs_val, group in df.groupby("HRS"):
            group = group.sort_values(by="Bits")
            plt.plot(group["Bits"], group["Misclassifications"], marker='o', linewidth=2.5, markersize=8, label=f'HRS = {hrs_val:.1e} $\Omega$')

        plt.title("Hardware Misclassifications vs. Bit Precision", fontsize=14, fontweight='bold')
        plt.xlabel("Quantization Bit Width", fontsize=12)
        plt.ylabel("Total Misclassifications", fontsize=12)
        
        plt.xticks(sorted(df["Bits"].unique()))
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(title="HRS Mean Variability", fontsize=10)
        plt.tight_layout()
        
        output_plot = "misclassifications_vs_bits_hrs.pdf"
        plt.savefig(output_plot, dpi=300)
        plt.close()
        
        print(f"\nUpdated aggregate plot generated successfully: {output_plot}")
    else:
        print("\nNo valid data found to process or plot.")
