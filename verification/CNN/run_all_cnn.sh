#!/bin/bash

echo "========================================================================================="
echo " PERFORMING MASTER CLEANUP: Removing old logs, data folders, and plot metrics..."
echo "========================================================================================="
rm -rf logs_cnn_hrs_*
rm -rf mnist_cnn_data_*
rm -f cnn_misclassification_data.csv
rm -f cnn_misclassifications_vs_bits_hrs.png
rm -f cnn_ideal_vs_hw_vs_bits_hrs.png cnn_metrics_breakdown.png
rm -f sim.vvp python_to_verilog.txt verilog_to_python.txt

MAX_PARALLEL_JOBS=30
HRS_VALUES=(5.0e4 5.0e5 5.0e19)
BIT_WIDTHS=(4 8 16 32)
SEED=0                      # CHANGE: fixed seed -> same images / reproducible training

export OMP_NUM_THREADS=1

# CHANGE: loop order swapped. Weights + images are generated ONCE per bit width and then reused
# for every HRS value. Before, the generator ran inside the HRS loop, so every (HRS, bits) point
# used a different network, a different image batch, and left stale image files behind.
for BITS in "${BIT_WIDTHS[@]}"; do
    echo "========================================================================================="
    echo " GENERATING WEIGHTS + 1000 IMAGES FOR ${BITS}-BIT (seed ${SEED})"
    echo "========================================================================================="

    python3 cnn_weight_image_generator.py $BITS $SEED
    if [ $? -ne 0 ]; then
        echo "ERROR: generator failed for ${BITS}-bit. Aborting."
        exit 1
    fi

    # CHANGE: verify there is exactly one integer image file per index (catches stale files)
    N_INT=$(ls mnist_cnn_data_${BITS}/sample_img_*_label_*.txt 2>/dev/null | wc -l)
    if [ "$N_INT" -ne 1000 ]; then
        echo "ERROR: expected 1000 integer images in mnist_cnn_data_${BITS}, found ${N_INT}. Aborting."
        exit 1
    fi

    for HRS in "${HRS_VALUES[@]}"; do
        echo "========================================================================================="
        echo " INITIALIZING CNN HARDWARE PIPELINE FOR ${BITS}-BIT PRECISION | HRS: ${HRS}"
        echo "========================================================================================="

        LOG_DIR="logs_cnn_hrs_${HRS}_${BITS}bit"
        mkdir -p "$LOG_DIR"

        echo "Launching 1000 CNN hardware inferences (Throttled to $MAX_PARALLEL_JOBS parallel threads)..."

        for i in {0..999}; do
            python3 -u isolated_parallel_cnn_inference.py $i $BITS $HRS > "$LOG_DIR"/log_img_${i}.txt 2>&1 &

            # Throttle execution to MAX_PARALLEL_JOBS concurrent processes
            while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL_JOBS ]; do
                sleep 0.5
            done
        done

        # Monitor remaining jobs to completion
        while [ $(jobs -r | wc -l) -gt 0 ]; do
            active_jobs=$(jobs -r | wc -l)
            echo -ne "  Active Background CNN Inferences Running: ${active_jobs} / 1000 ...\r"
            sleep 2
        done

        echo -e "\n========================================================================================="
        echo "  ${BITS}-BIT & HRS ${HRS} CNN SIMULATIONS COMPLETE!"
        echo "  Generating Plot with evaluate_cnn_metrics.py..."
        echo "========================================================================================="

        python3 evaluate_cnn_metrics.py

        sleep 3
    done
done

echo "ALL SCHEDULED CNN PIPELINES COMPLETED SUCCESSFULLY."
