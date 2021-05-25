/*
 Copyright (c) 2020, Xilinx
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither the name of FINN nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

module rd_control #(
    parameter NPIXELS = 1024,
    parameter WORDS_PER_PX = 1,
    parameter IFMWidth = 8,
    parameter STRIDE = 1,
    parameter MMV_IN = 2,
    parameter KERNEL_HEIGHT = 3,
    parameter KERNEL_WIDTH = 3,
    parameter OFMWidth = 6,
    parameter OFMHeight = 6,
    parameter PADDING_WIDTH = 0,
    parameter PADDING_HEIGHT =0,
    parameter BUFFER_DEPTH = 20
)
(
    input aclk,
    input aresetn,

    input wr_handshake,
    input [$clog2(BUFFER_DEPTH/MMV_IN)-1:0] wr_addr,

    output ready,
    input handshake,
    output done,
    input full,

    output reg [$clog2(BUFFER_DEPTH)-1:0] addr,
    output en,
    output enq,
    output valid
);

reg buffer_empty = 0;
reg buffer_empty_i = 0;
reg buffer_empty_ii = 0;

reg r_valid, q_valid;
wire enaB, enaB_q;
reg op_axis_tready_del;

reg [$clog2(KERNEL_HEIGHT) - 1: 0] kh = 0;
reg [$clog2(KERNEL_WIDTH) - 1 : 0] kw= 0;
reg [$clog2(OFMWidth) - 1: 0] ofm_column_tracker = 0;
reg [$clog2(OFMHeight) - 1: 0]ofm_row_tracker = 0;
reg [$clog2(WORDS_PER_PX) : 0] channel_tracker = 0;
wire weA;
reg [$clog2(MMV_IN) - 1: 0] mmv_col_tracker = 0;
reg [$clog2(MMV_IN) - 1: 0] mmv_row_tracker = 0;

reg [$clog2(MMV_IN) - 1: 0] mmv_col_tracker_advance = 0;
reg [$clog2(MMV_IN) - 1: 0] mmv_row_tracker_advance = 0;

reg [$clog2(MMV_IN) - 1: 0] mmv_sub_tracker = 0;
reg [$clog2(OFMHeight *  IFMWidth * WORDS_PER_PX) : 0] starting_pos_i = 0;
reg [$clog2(BUFFER_DEPTH) - 1 : 0] starting_pos = 0;
reg [$clog2(BUFFER_DEPTH) - 1 : 0] pending_rd_cntr = 0;
//reg [$clog2(BUFFER_DEPTH+WORDS_PER_PX*(KERNEL_WIDTH+((KERNEL_HEIGHT-1)*IFMWidth)) + WORDS_PER_PX)-1 : 0] addr = 0;
reg mmvshift=0;

assign done = (buffer_empty && handshake);

//buffer read logic
assign valid = q_valid;
assign enaB = full & !buffer_empty_i & (enaB_q | ~r_valid) & (!(ofm_row_tracker != 0 && ofm_column_tracker == 0 && kh == 0 && kw == 0 && channel_tracker == 0) || (pending_rd_cntr == 0));
assign enaB_q = (ready | ~q_valid);

wire rd_cntr_incdec;

generate if (MMV_IN > 1) begin: use_immv
    assign rd_cntr_incdec = (ofm_column_tracker != 0) && ((mmv_col_tracker == MMV_IN - 1 && kh == 0 && kw == KERNEL_WIDTH - 1) || ofm_column_tracker == OFMWidth - 1 && kh == KERNEL_HEIGHT - 1 && kw == 0);      
end else begin: no_immv
    assign rd_cntr_incdec = (kh == 0 && kw == KERNEL_WIDTH - 1) || (ofm_column_tracker == OFMWidth - 1 && kh == KERNEL_HEIGHT - 1 && kw < KERNEL_WIDTH - 1);
end
endgenerate

always @(posedge aclk)
    if(~aresetn | done) begin
        pending_rd_cntr <= BUFFER_DEPTH/MMV_IN;
    end else if(wr_handshake & !rd_cntr_incdec)
        pending_rd_cntr <= pending_rd_cntr - 1;
    else if( !wr_handshake & rd_cntr_incdec)
        pending_rd_cntr <= pending_rd_cntr + 1;  

always @(posedge aclk)
    if(~aresetn)
        q_valid <= 0;
    else if(enaB_q)
        q_valid <= r_valid & ~buffer_empty_i;
        
always @(posedge aclk)
    if(~aresetn)
        r_valid <= 0;
    else if(enaB_q | ~r_valid)
        r_valid <=  enaB;        

always @(posedge aclk) begin
    if (~aresetn) begin
        buffer_empty_i <= 0;
        buffer_empty_ii <= 0;
    end else begin
        buffer_empty_i <= buffer_empty;
        buffer_empty_ii <= buffer_empty_i;
    end
end

always @(posedge aclk) begin
    if (~aresetn) begin
        op_axis_tready_del <= 0;
    end else begin
        op_axis_tready_del <= ready;
    end
end
  
//2
always @(posedge aclk) begin
    if (~aresetn | done) begin
        buffer_empty <= 0;
    end else if (kh==KERNEL_HEIGHT-1 && kw==KERNEL_WIDTH-1 && ofm_row_tracker == OFMHeight - 1 && ofm_column_tracker == OFMWidth-1 && channel_tracker == WORDS_PER_PX -1) begin 
        buffer_empty <= 1;
    end
end
// process to read data
//3
always @(*) begin
    if (~aresetn | done) begin
        addr = 0;
    end else begin
        if(MMV_IN == 1) begin
            addr = starting_pos + kw * WORDS_PER_PX + kh*(IFMWidth * WORDS_PER_PX) + channel_tracker; 
        end else begin
            addr = starting_pos + kw + MMV_IN * mmvshift + kh * (IFMWidth * WORDS_PER_PX) + (channel_tracker * MMV_IN); 
        end
        if(addr >=BUFFER_DEPTH) begin
            addr = addr - BUFFER_DEPTH;
        end
    end
end

//6
always @(posedge aclk) begin
    if (~aresetn) begin
        channel_tracker <= 0;
    end else if (done) begin
        channel_tracker <= 0;   
    end else begin
        if ((full || wr_addr == (BUFFER_DEPTH/MMV_IN) - 1  ) & enaB) begin
            if ((channel_tracker < WORDS_PER_PX - 1)) begin
                channel_tracker <= channel_tracker + 1;
            end else begin
                channel_tracker <= 0;
            end
        end
    end
end


//7
always @(posedge aclk) begin
    if (~aresetn) begin
        kw <= 0;
        kh <= 0;
        mmv_sub_tracker <= 0;
    end else if (done) begin
        kw <= 0;
        kh <= 0;
        mmv_sub_tracker <= 0;    
    end else
    if (full & enaB && (channel_tracker == WORDS_PER_PX - 1)) begin
        if ((kw < KERNEL_WIDTH - 1)) begin
            kw <= kw + 1;
            if (mmv_sub_tracker < MMV_IN - 1) begin
                mmv_sub_tracker <= mmv_sub_tracker + 1;
            end else begin
                mmv_sub_tracker <= 0;
            end
        end else if (kw == KERNEL_WIDTH - 1) begin
            kw <= 0;
            mmv_sub_tracker <= mmv_col_tracker_advance;
            if (kh < (KERNEL_HEIGHT - 1) )begin
                kh <= kh + 1;
            end else begin
                kh <= 0;
            end
        end
    end
end

//8
always @(posedge aclk) begin
    if(~aresetn | done) begin
        ofm_column_tracker <= 0;
        ofm_row_tracker <= 0;
    end else begin
        if (full && enaB && (channel_tracker == WORDS_PER_PX - 1 )) begin
            if(kw==KERNEL_WIDTH-1 && kh==KERNEL_HEIGHT-1) begin
                if(ofm_column_tracker < (OFMWidth - 1)) begin
                    ofm_column_tracker <= ofm_column_tracker + 1;
                    if (mmv_col_tracker < MMV_IN - 1) begin
                        mmv_col_tracker <= mmv_col_tracker + 1;
                    end else begin
                        mmv_col_tracker <= 0;
                    end
                end else begin
                    ofm_column_tracker <= 0;
                    mmv_col_tracker <= 0;
                    if (ofm_row_tracker < (OFMHeight - 1)) begin
                        ofm_row_tracker <= ofm_row_tracker + 1;
                        if (mmv_row_tracker < MMV_IN - 1) begin
                            mmv_row_tracker <= mmv_row_tracker + 1;
                        end else begin
                            mmv_row_tracker <= 0;
                        end
                    end else begin
                        ofm_row_tracker <= 0;
                        mmv_row_tracker <= 0;
                    end
                end
            end
        end
    end
end

//9
always @(posedge aclk) begin
    if(~aresetn | done) begin
        mmv_col_tracker_advance <= 0;
    end else begin
        if (full && enaB ) begin
            if ((kw+1)*WORDS_PER_PX+channel_tracker == KERNEL_WIDTH * WORDS_PER_PX - 1) begin
                if (kh == KERNEL_HEIGHT - 1) begin
                    if(ofm_column_tracker < (OFMWidth - 1)) begin
                        if (mmv_col_tracker < MMV_IN - 1) begin
                            mmv_col_tracker_advance <= mmv_col_tracker_advance + 1;
                        end else begin
                            mmv_col_tracker_advance <= 0;
                        end
                    end else begin
                        mmv_col_tracker_advance <= 0;
                    end
                end
            end
        end  
    end            
end

always @(posedge aclk) begin
    if(~aresetn | done) begin
        mmv_row_tracker_advance <= 0;
    end else begin
        if (full && enaB ) begin
            if ((kw+1)*WORDS_PER_PX+channel_tracker == KERNEL_WIDTH * WORDS_PER_PX - 1) begin
                if (kh == KERNEL_HEIGHT - 1) begin
                    if(ofm_column_tracker < (OFMWidth - 1)) begin
                        if (mmv_col_tracker < MMV_IN - 1) begin
                            mmv_row_tracker_advance <= mmv_col_tracker_advance + 1;
                        end else begin
                            mmv_row_tracker_advance <= 0;
                        end
                    end else begin
                        mmv_row_tracker_advance <= 0;
                end
            end
            end
        end
    end
end
//10
always @(posedge aclk) begin
    if (~aresetn | done) begin
        starting_pos <= 0;
    end else begin
        if(starting_pos_i >= BUFFER_DEPTH) begin
            starting_pos <= starting_pos_i - BUFFER_DEPTH;
        end else begin
            starting_pos <= starting_pos_i;
        end
    end
end

//11
always @(posedge aclk) begin
    if (~aresetn | done) begin
        starting_pos_i <= 0;
    end else begin
        if (full & enaB) begin
            if(kh*KERNEL_WIDTH*WORDS_PER_PX+kw*WORDS_PER_PX+channel_tracker+1==KERNEL_WIDTH*KERNEL_HEIGHT*WORDS_PER_PX - 1) begin
                if ((ofm_column_tracker < (OFMWidth - 1 )) && (ofm_column_tracker >= PADDING_WIDTH)) begin //should not increment by one if it's less than padding_width or greater than End-width - padding-width
                    if (MMV_IN > 1) begin
                        if (mmv_col_tracker != MMV_IN - 1 || WORDS_PER_PX == 1) begin
                            starting_pos_i <= starting_pos+(STRIDE);
                        end else begin
                            starting_pos_i <= starting_pos+(STRIDE)+MMV_IN;
                        end
                    end else begin
                        starting_pos_i = starting_pos + STRIDE * WORDS_PER_PX;          
                    end 
                end
                else if (ofm_column_tracker == OFMWidth - 1) begin
                    if ((ofm_row_tracker >= PADDING_HEIGHT)) begin
                        //starting_pos_i <= starting_pos + (KERNEL_WIDTH-PADDING_WIDTH)*(WORDS_PER_PX) + (STRIDE - 1) * (IFMWidth * WORDS_PER_PX);
                        starting_pos_i <= (ofm_row_tracker + 1) * STRIDE * IFMWidth * WORDS_PER_PX; 
                    end else begin
                        starting_pos_i <= 0;
                    end               
                end else if(ofm_column_tracker < PADDING_WIDTH) begin
                    starting_pos_i <= starting_pos;
                end 
            end
        end
    end
end

//12
always @(posedge aclk) begin
    if (~aresetn | done) begin 
        mmvshift <= 0;
    end else begin
        if (mmv_sub_tracker==MMV_IN-1 && channel_tracker == WORDS_PER_PX - 1 && WORDS_PER_PX > 1) begin
            mmvshift <= 1;
        end else if (kw == KERNEL_WIDTH - 1 && channel_tracker == WORDS_PER_PX - 1)begin
            mmvshift <= 0;
        end
    end
end

endmodule
