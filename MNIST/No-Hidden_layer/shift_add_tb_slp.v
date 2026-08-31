/*module shift_add_tb;

    integer file_in;
    integer file_out;
    integer scan_status; 
    
    integer col_idx;
    integer shift_amount;
    integer max_col_seen;
    integer i;

    // 1. CHANGED FROM INTEGER/REG TO REAL (Floating-Point)
    real current_in;
    real accumulators [0:255];
    
    reg column_active [0:255]; 

    initial begin
        max_col_seen = 0;
        
        // Zero out memory
        for (i = 0; i < 256; i = i + 1) begin
            // 2. Initialize reals with 0.0 instead of 64'd0
            accumulators[i] = 0.0;
            column_active[i] = 1'b0;
        end
        
        // Open the file written by Python
        file_in = $fopen("python_to_verilog.txt", "r");
        if (file_in == 0) begin
            $display("Error: Could not open python_to_verilog.txt.");
            $finish;
        end
        
        // --- THE FOOLPROOF READ LOOP ---
        
        // 3. Changed the read format to "%f" (Float) for the current
        scan_status = $fscanf(file_in, "%d %f %d\n", col_idx, current_in, shift_amount);
        
        while (scan_status == 3) begin
            
            // 4. THE MATH CHANGE: You cannot bit-shift a 'real' number. 
            // Instead, we multiply the decimal by (2^shift_amount).
            // In Verilog math, (1 << shift_amount) generates that exact multiplier.
            //accumulators[col_idx] = accumulators[col_idx] + (current_in * (1 << shift_amount));
            accumulators[col_idx] = accumulators[col_idx] + (current_in * (64'd1 << shift_amount));
            column_active[col_idx] = 1'b1;
            
            if (col_idx > max_col_seen) begin
                max_col_seen = col_idx;
            end
            
            // Read the next line
            scan_status = $fscanf(file_in, "%d %f %d\n", col_idx, current_in, shift_amount);
        end
        
        $fclose(file_in);

        // Write final mathematical results out
        file_out = $fopen("verilog_to_python.txt", "w");
        for (i = 0; i <= max_col_seen; i = i + 1) begin
            if (column_active[i] == 1'b1) begin
                // 5. Changed the write format to "%f" to output the final decimal sum
                $fdisplay(file_out, "%d %e", i, accumulators[i]);
            end
        end
        
        $fclose(file_out);
        $finish;
    end

endmodule*/
module shift_add_tb;
integer file_in;
integer file_out;
integer scan_status; 

integer col_idx;
integer shift_amount;
integer max_col_seen;
integer i;

// Keep these as real so we don't lose precision or hit overflow
real accumulators [0:255];
reg column_active [0:255]; 

real current_in;

initial begin
    max_col_seen = 0;
    
    for (i = 0; i < 256; i = i + 1) begin
        accumulators[i] = 0.0;
        column_active[i] = 1'b0;
    end
    
    file_in = $fopen("python_to_verilog.txt", "r");
    if (file_in == 0) begin
        $display("Error: Could not open python_to_verilog.txt.");
        $finish;
    end
    
    // Scan inputs as reals
    scan_status = $fscanf(file_in, "%d %f %d\n", col_idx, current_in, shift_amount);
    
    while (scan_status == 3) begin
        // Accumulate using real-number math (handles negatives perfectly)
        accumulators[col_idx] = accumulators[col_idx] + (current_in * (64'd1 << shift_amount));
        column_active[col_idx] = 1'b1;
        
        if (col_idx > max_col_seen) max_col_seen = col_idx;
        
        scan_status = $fscanf(file_in, "%d %f %d\n", col_idx, current_in, shift_amount);
    end
    
    $fclose(file_in);

    file_out = $fopen("verilog_to_python.txt", "w");
    for (i = 0; i <= max_col_seen; i = i + 1) begin
        if (column_active[i] == 1'b1) begin
            // Use native %e (Scientific Notation) format which is perfectly supported
            // for real data types in Verilog and easily parsed by Python.
            $fdisplay(file_out, "%d %e", i, accumulators[i]);
        end
    end
    
    $fclose(file_out);$finish;
end


endmodule
