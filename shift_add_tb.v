module shift_add_tb;

    integer file_in;
    integer file_out;
    integer scan_status; // Added this back to hold the read result!
    
    integer col_idx;
    integer current_in;
    integer shift_amount;
    integer max_col_seen;
    integer i;

    // Memory arrays explicitly declared OUTSIDE the initial block
    reg [63:0] accumulators [0:255];
    reg column_active [0:255]; 

    initial begin
        max_col_seen = 0;
        
        // Zero out memory
        for (i = 0; i < 256; i = i + 1) begin
            accumulators[i] = 64'd0;
            column_active[i] = 1'b0;
        end
        
        // Open the file written by Python
        file_in = $fopen("python_to_verilog.txt", "r");
        if (file_in == 0) begin
            $display("Error: Could not open python_to_verilog.txt.");
            $display("%d",file_in);
			$finish;
        end
		
        // --- THE FOOLPROOF READ LOOP ---
        
        // 1. Read the very first line BEFORE the loop starts
        scan_status = $fscanf(file_in, "%d %d %d\n", col_idx, current_in, shift_amount);
        
        // 2. Only run the loop if we successfully read exactly 3 numbers
        while (scan_status == 3) begin
            
            // Do the math
            accumulators[col_idx] = accumulators[col_idx] + (current_in << shift_amount);
            column_active[col_idx] = 1'b1;
            
            if (col_idx > max_col_seen) begin
                max_col_seen = col_idx;
            end
            
            // 3. Read the next line at the very bottom of the loop
            scan_status = $fscanf(file_in, "%d %d %d\n", col_idx, current_in, shift_amount);
        end
        
        // -------------------------------
        
        $fclose(file_in);

        // Write final mathematical results out
        file_out = $fopen("verilog_to_python.txt", "w");
        for (i = 0; i <= max_col_seen; i = i + 1) begin
            if (column_active[i] == 1'b1) begin
                $fdisplay(file_out, "%d %d", i, accumulators[i]);
            end
        end
        
        $fclose(file_out);
        $finish;
    end

endmodule