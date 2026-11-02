#!/bin/bash

MAX_PARALLEL_JOBS=60
HRS_VALUES=(5.0e4 5.0e5 5.0e19)
#HRS_VALUES=(5e19)
BIT_WIDTHS=(4 8 16 32)
#BIT_WIDTHS=(4 8 16)
for HRS in "${HRS_VALUES[@]}"; do
    for BITS in "${BIT_WIDTHS[@]}"; do
        echo "========================================================================================="
        echo " INITIALIZING HARDWARE PIPELINE FOR ${BITS}-BIT PRECISION | HRS: ${HRS}"
        echo "========================================================================================="
        
        LOG_DIR="logs_hrs_${HRS}_${BITS}bit"
        mkdir -p "$LOG_DIR"
        rm -f "$LOG_DIR"/log_img_*.txt

        echo "Generating weights and 1000 images for ${BITS}-bit..."
        python3 mlp_weight_image_generator.py $BITS

        echo "Launching 1000 hardware inferences (Throttled to $MAX_PARALLEL_JOBS parallel threads)..."

        for i in {0..9}; do
          export OMP_NUM_THREADS=1
          python3 -u with_LR_isolated_parallel_inference_pipeline_3.py $i $BITS $HRS > "$LOG_DIR"/log_img_${i}.txt 2>&1 &

          # Throttle execution to a maximum of 60 concurrent processes
          while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL_JOBS ]; do
              sleep 0.1
          done
        done

        # Monitor remaining jobs to completion
        while [ $(jobs -r | wc -l) -gt 0 ]; do
            active_jobs=$(jobs -r | wc -l)
            echo -ne "  Active Background Inferences Running: ${active_jobs} / 1000 ...\r"
            sleep 2
        done

        echo -e "\n========================================================================================="
        echo "  ${BITS}-BIT & HRS ${HRS} SIMULATIONS COMPLETE!"
        echo "  Generating Plot with evaluate_metrics.py..."
        echo "========================================================================================="

        python3 evaluate_metrics.py "$LOG_DIR" "$BITS" "$HRS"
        
        sleep 3
    done
done

echo "ALL SCHEDULED PIPELINES COMPLETED SUCCESSFULLY.
