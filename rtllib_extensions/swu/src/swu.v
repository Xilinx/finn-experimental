`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 05/05/2021 10:19:13 AM
// Design Name: 
// Module Name: mmv_input_swu
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module swu #(
    parameter SIMD = 32,
    parameter STRIDE_HT = 1,
    parameter STRIDE_WT = 1,
    parameter IFMChannels = 128,
    parameter KERNEL_HEIGHT = 3,
    parameter KERNEL_WIDTH = 3,
    parameter RAM_STYLE = "auto",

	parameter IFMWidth = 5,
	parameter IFMHeight = 5,
	parameter PADDING_WIDTH = 0,
	parameter PADDING_HEIGHT = 0,
	parameter OFMWidth = 3,
	parameter OFMHeight = 3,

	//depths per stream
	parameter IP_PRECISION = 1,
	parameter MMV_IN = 1,
	parameter MMV_OUT = 1,
	parameter DWS = 0,
	parameter ZEROPAD = 0)


(
input aclk,
input aresetn,

input [MMV_IN * SIMD * IP_PRECISION - 1 : 0] s_axis_tdata,
input s_axis_tvalid,
output s_axis_tready,

output [MMV_OUT * SIMD * IP_PRECISION - 1 : 0] m_axis_tdata,
input m_axis_tready,
output m_axis_tvalid

    );
    
localparam   BUFFER_SIZE = (STRIDE_HT - 1 + KERNEL_HEIGHT) * IFMWidth * IFMChannels/SIMD ;
localparam floor_O_BY_I = MMV_OUT/MMV_IN;
localparam ceil_O_BY_I = (MMV_OUT + MMV_IN-1)/MMV_IN;

localparam EFF_CHANNELS = IFMChannels/SIMD;
localparam SIZEA = BUFFER_SIZE/MMV_IN;
localparam SIZEB = BUFFER_SIZE;
localparam ADDRWIDTHA = $clog2(SIZEA);
localparam ADDRWIDTHB = $clog2(SIZEB);
localparam WIDTHA = MMV_IN * SIMD * IP_PRECISION ;
localparam WIDTHB = SIMD * IP_PRECISION;
localparam ALLOW_WRITES = ((KERNEL_HEIGHT - PADDING_HEIGHT - 1) * IFMWidth/MMV_IN + KERNEL_WIDTH + ((MMV_OUT - 1) * STRIDE_WT)) * EFF_CHANNELS;
localparam EXTRA_ROW = (((IFMWidth%2 == 0) & PADDING_HEIGHT == 0)) || ((IFMWidth%2 != 0) & (PADDING_HEIGHT != 0)) ? 1 : 0;
//(* ram_style = RAM_STYLE *) reg [SIMD*IP_PRECISION-1:0] mem[BUFFER_SIZE - 1:0];  

integer counter=0;   
reg buffer_full=0;
reg buffer_full_i=0;
reg start_write = 0;

reg buffer_empty = 0;
reg buffer_empty_i = 0;
reg buffer_empty_ii = 0;

reg r_valid, q_valid;
wire enaB, enaB_q, enaB_r;
reg enaB_reg = 0;
reg [$clog2(IFMHeight * IFMWidth) - 1 : 0] input_pixel= 0;
reg [$clog2(KERNEL_HEIGHT) - 1: 0] kh = 0;
reg [$clog2(KERNEL_WIDTH) - 1 : 0] kw= 0;
reg [$clog2(KERNEL_HEIGHT) - 1: 0] kh_tracker[MMV_OUT - 1 : 0] ;
reg [$clog2(KERNEL_WIDTH) - 1 : 0] kw_tracker[MMV_OUT - 1 : 0];
reg [$clog2(OFMWidth) - 1: 0] ofm_column_tracker[MMV_OUT - 1 :0];
reg [$clog2(OFMHeight) - 1: 0]ofm_row_tracker[MMV_OUT - 1 :0];
reg [$clog2(EFF_CHANNELS) : 0] ch_ptr = 0;
wire weA;
reg [$clog2(MMV_IN) - 1: 0] mmv_col_tracker[MMV_OUT - 1 :0];
reg [$clog2(KERNEL_HEIGHT*STRIDE_HT) - 1: 0] kernel_row_tracker = 0;

reg [$clog2(MMV_IN) - 1: 0] mmv_col_tracker_advance[MMV_OUT - 1 : 0] ;
reg [$clog2(MMV_IN) - 1: 0] mmv_row_tracker_advance ;

reg [$clog2(MMV_IN) - 1: 0] mmv_sub_tracker [MMV_OUT - 1 : 0];
reg [$clog2(OFMHeight *  IFMWidth * EFF_CHANNELS) : 0] starting_pos_i [MMV_OUT  - 1 :0] ;
reg [$clog2(BUFFER_SIZE) - 1 : 0] starting_pos[MMV_OUT  - 1 :0];
reg [$clog2(BUFFER_SIZE+EFF_CHANNELS*(KERNEL_WIDTH+((KERNEL_HEIGHT-1)*IFMWidth)) + EFF_CHANNELS)-1 : 0] pos [MMV_OUT - 1:0];
reg mmvshift[MMV_OUT - 1 : 0];
reg [$clog2(BUFFER_SIZE/MMV_IN) : 0] total_pending_rds;

reg [$clog2(BUFFER_SIZE/MMV_IN) - 1 : 0] crtcl_rd_cntr[STRIDE_HT - 1: 0];
reg [$clog2(BUFFER_SIZE/MMV_IN) - 1 : 0] crtcl_rd_cntr_del;

reg [$clog2(BUFFER_SIZE/MMV_IN) - 1 : 0] pending_rd_cntr;
reg [$clog2(BUFFER_SIZE/MMV_IN) - 1 : 0] new_rd_cntr;
reg [$clog2(IFMWidth * IFMHeight * EFF_CHANNELS)-1:0] input_counter;
reg finish_rds;
wire m_axis_hs;
wire s_axis_hs;
wire restart;
reg stride_toggle;
//reg [log2(MMV_OUT) - 1:0] m;
//reg [log2(MMV_OUT) - 1:0] n;
//reg [log2(MMV_OUT) - 1:0] o;
integer m , n, o;
assign s_axis_tready = total_pending_rds != 0;
assign weA = s_axis_hs;
assign ram_enq = m_axis_tready | ~q_valid;
assign m_axis_tvalid = q_valid;
assign enaB = start_write & !buffer_empty_i 
            & (enaB_q | ~r_valid) 
            & ((!ofm_row_tracker[0] <= PADDING_HEIGHT && ofm_column_tracker[0] > 0 && kh == KERNEL_HEIGHT - 2 && kw == KERNEL_WIDTH - 1 && ch_ptr == 0) 
            || (buffer_full || counter >= ALLOW_WRITES + (ofm_row_tracker[0]*IFMWidth/MMV_IN) + (ofm_column_tracker[MMV_OUT - 1] * EFF_CHANNELS) * STRIDE_WT))
            & (!(ofm_row_tracker[0] != 0 && ofm_column_tracker[0] > 0 && kh == KERNEL_HEIGHT - 2 && kw == KERNEL_WIDTH - 1 && ch_ptr == 0) 
            ||(crtcl_rd_cntr[0] * MMV_IN <= (IFMWidth * EFF_CHANNELS - KERNEL_WIDTH * EFF_CHANNELS - (ofm_column_tracker[MMV_OUT-1]*EFF_CHANNELS * STRIDE_WT))) & buffer_full)
            & (!(ofm_row_tracker[0] != 0 && ofm_column_tracker[0] == 0 && kh == KERNEL_HEIGHT - 2  && ch_ptr == 0) 
            || (crtcl_rd_cntr[0] * MMV_IN < (IFMWidth * EFF_CHANNELS - ((kw + 1) * EFF_CHANNELS) - ofm_column_tracker[MMV_OUT - 1])) & buffer_full);

assign enaB_q = (m_axis_tready | ~q_valid);
assign enaB_r = start_write & !buffer_empty_i & (m_axis_tready | ~r_valid);


wire[MMV_OUT - 1 : 0] zeropad;
reg zeropad_reg[MMV_OUT - 1 : 0] ;
genvar mi;
generate
  for (mi = 0; mi < MMV_OUT; mi = mi + 1) begin : MI_SLOT
    asymmetrc_ram #(
   .SIZEA(SIZEA),
   .SIZEB(SIZEB),
   .WIDTHA(WIDTHA),
   .WIDTHB(WIDTHB),
   .ADDRWIDTHA(ADDRWIDTHA),
   .ADDRWIDTHB(ADDRWIDTHB),
   .RAM_STYLE(RAM_STYLE)
)
    ram (
   .clkA(aclk),
   .clkB(aclk),
   .addrA(counter),
   .addrB(pos[mi]),
   .diA(s_axis_tdata),
   .doB(m_axis_tdata[(mi+1)*SIMD*IP_PRECISION-1 -: SIMD*IP_PRECISION]),
   .enaA(1),
   .enaB(enaB),
   .enaB_q(enaB_q),
   .weA(weA),
   .zeropad(zeropad_reg[mi])
);
  end
endgenerate


assign m_axis_hs = m_axis_tready && m_axis_tvalid;
assign s_axis_hs = s_axis_tready && s_axis_tvalid;
assign restart = m_axis_tready && m_axis_tvalid && buffer_empty;

wire inc_pending_rd_gen;
wire inc_pending_rd_last;
wire [$clog2(MMV_OUT * STRIDE_WT + 1) - 1 : 0] inc_rd_amt;
wire [$clog2(EFF_CHANNELS * ((IFMWidth - (OFMWidth - PADDING_WIDTH)*STRIDE_WT) + 1))-1:0] inc_rd_amt_last;

wire inc_ch_ptr;
wire valid_ptr_vals;
wire inc_kw;
wire inc_pending_rd_stride;
reg s_axis_hs_reg;
assign valid_ptr_vals = start_write & enaB;

wire start_new_row;
assign start_new_row  = valid_ptr_vals && (ch_ptr == EFF_CHANNELS - 1 ) && (kw==KERNEL_WIDTH-1 && kh==KERNEL_HEIGHT-1) && (ofm_column_tracker[0] >= OFMWidth - MMV_OUT);

generate if (MMV_IN > 1) begin : inc_immv
   assign inc_rd_amt = 1;
end else begin : inc_noimmv
   assign inc_rd_amt = MMV_OUT * STRIDE_WT;
end
endgenerate

generate if (MMV_IN == 1) begin: inc_no_immv
   assign inc_rd_amt_last = (IFMWidth - (OFMWidth - PADDING_WIDTH) * STRIDE_WT) * EFF_CHANNELS;
end else begin : inc_eff_channels
   assign inc_rd_amt_last = EFF_CHANNELS;
end
endgenerate

generate if (STRIDE_HT > 1) begin : inc_stride
   assign inc_pending_rd_stride = ofm_column_tracker[0] == OFMWidth - MMV_OUT && kh == KERNEL_HEIGHT - 1 && kw == KERNEL_HEIGHT - 1  && ch_ptr == EFF_CHANNELS - 1 &&  ofm_row_tracker[0] >= PADDING_HEIGHT && ofm_row_tracker[0] < OFMHeight - 1 - PADDING_HEIGHT - 1 + EXTRA_ROW ;
end else begin : inc_nostride
   assign inc_pending_rd_stride = 0;
end
endgenerate

genvar z;
generate if (ZEROPAD == 0) begin : zeropad_gen
   assign zeropad = 0;
end else begin : self_pad
   for( z = 0; z < MMV_OUT; z = z + 1) begin
   assign zeropad[z] = (ofm_row_tracker[z] < PADDING_HEIGHT && kh < PADDING_HEIGHT) || (ofm_row_tracker[z] >= OFMHeight - PADDING_HEIGHT && kh >= KERNEL_HEIGHT - PADDING_HEIGHT) || (ofm_column_tracker[z] < PADDING_WIDTH && kw < PADDING_WIDTH) || (ofm_column_tracker[z] >= OFMWidth - PADDING_WIDTH && kw >= KERNEL_WIDTH - PADDING_WIDTH);
   end
end
endgenerate

generate if (DWS == 0) begin : normal_cnv_kw
   assign inc_kw = valid_ptr_vals && (ch_ptr == EFF_CHANNELS - 1);
end else begin : dws_cnv_kw
   assign inc_kw = valid_ptr_vals;
end
endgenerate

generate if (DWS == 0) begin : normal_cnv_ch
   assign inc_ch_ptr = valid_ptr_vals ;
end else begin : dws_cnv_ch
   assign inc_ch_ptr = valid_ptr_vals && (kw == KERNEL_WIDTH - 1 && kh == KERNEL_HEIGHT - 1);
end
endgenerate

// this does not depend on padding yet
generate if (MMV_IN > 1) begin : use_immv
   assign inc_pending_rd_gen = ((ofm_column_tracker[0] >= PADDING_WIDTH
                              && ofm_row_tracker[0] < OFMHeight - 1 - PADDING_HEIGHT) 
                              && enaB_reg 
                              && (ofm_row_tracker[0] >= PADDING_HEIGHT || (STRIDE_HT > PADDING_HEIGHT))) 
                              && (mmv_col_tracker[0] + STRIDE_WT * MMV_OUT >= MMV_IN && kh == 1 && kw == KERNEL_WIDTH - 1) ;
end else begin : no_immv
   assign inc_pending_rd_gen = ((ofm_column_tracker[0] >= PADDING_WIDTH 
                              && ofm_row_tracker[0] < OFMHeight - 1 - PADDING_HEIGHT) 
                              && enaB_reg 
                              && ((ofm_row_tracker[0] >= PADDING_HEIGHT) || (STRIDE_HT > PADDING_HEIGHT))) 
                              && (kh == 0 && kw == KERNEL_WIDTH - 1) ;
end 
endgenerate


// according to my intuition. for no immv, this is what must be done, but i will leave it for now
generate if(MMV_IN > 1) begin: use_immv1
assign inc_pending_rd_last =  ofm_column_tracker[0] == OFMWidth - MMV_OUT 
                           && ofm_row_tracker[0] < OFMHeight - 1 - PADDING_HEIGHT 
                           && (ofm_row_tracker[0] >= PADDING_HEIGHT || (STRIDE_HT > PADDING_HEIGHT))
                           && enaB_reg
                           && kh == KERNEL_HEIGHT - 1
                           && kw == 0
                           && ((mmv_col_tracker[0] + STRIDE_WT*MMV_OUT < MMV_IN) || MMV_OUT > 1)
                           && ch_ptr == EFF_CHANNELS - 1;
end else begin : no_immv1
assign inc_pending_rd_last = ofm_column_tracker[0] == OFMWidth - MMV_OUT 
                           && ofm_row_tracker[0] < OFMHeight - 1 - PADDING_HEIGHT 
                           && (ofm_row_tracker[0] >= PADDING_HEIGHT || (STRIDE_HT > PADDING_HEIGHT))
                           && enaB_reg
                           && kh == KERNEL_HEIGHT - 1
                           && kw == 0
                           && ch_ptr == EFF_CHANNELS - 1;
end
endgenerate
/*generate if (MMV_IN > 1 && (STRIDE == 1 || PADDING_WIDTH == 0)) begin : use_immv1
   assign inc_pending_rd_last = ((ofm_column_tracker[0] == OFMWidth - MMV_OUT  && ofm_row_tracker[0] < OFMHeight - STRIDE - PADDING_HEIGHT && kh == KERNEL_HEIGHT - 1 && kw == 0)) && enaB_reg && ofm_row_tracker[0] >= PADDING_HEIGHT;
end else if (STRIDE == 1 || PADDING_WIDTH == 0) begin : no_immv1
    assign inc_pending_rd_last = (ofm_column_tracker[0] == OFMWidth - MMV_OUT && kh == KERNEL_HEIGHT - 1 && kw < (KERNEL_WIDTH - 1 - PADDING_WIDTH)) && enaB_reg && ofm_row_tracker[0] >= PADDING_HEIGHT && ofm_row_tracker[0] < OFMHeight - STRIDE - PADDING_HEIGHT;
end else if (MMV_IN == 1 && STRIDE > 1 && PADDING_WIDTH > 0) begin : stride_and_padding_no_immv
   assign inc_pending_rd_last = ofm_column_tracker[0] <= PADDING_WIDTH && enaB_reg && (kh == 0 && kw == KERNEL_WIDTH - 1) && ofm_row_tracker[0] >= PADDING_HEIGHT && ofm_row_tracker[0] < OFMHeight - 1 - PADDING_HEIGHT;
end else if(STRIDE > 1 && MMV_IN > 1 && PADDING_WIDTH > 0) begin : stride_and_padding_and_mmv
   assign inc_pending_rd_last = ofm_column_tracker[0] == OFMWidth - MMV_OUT && ofm_row_tracker[0] < OFMHeight - STRIDE - PADDING_HEIGHT && enaB_reg && kh == KERNEL_HEIGHT - 1 && kw == KERNEL_WIDTH - 2 && (mmv_col_tracker[0] + STRIDE*MMV_OUT < MMV_IN) && ch_ptr == EFF_CHANNELS - 1;   
end else if (STRIDE > 1 && MMV_IN > 1) begin : stride_and_padding_and_mmv
   assign inc_pending_rd_last = ofm_column_tracker[0] == OFMWidth - MMV_OUT && kh == KERNEL_HEIGHT - 1 && kw == 0  && ofm_row_tracker[0] >= PADDING_HEIGHT && ofm_row_tracker[0] < OFMHeight - 1 - PADDING_HEIGHT;
end
endgenerate*/

wire [$clog2(KERNEL_HEIGHT*KERNEL_WIDTH*EFF_CHANNELS)-1:0]inc_start_pos;
generate if (DWS == 0) begin : normal_cnv_st
   assign inc_start_pos = kh * KERNEL_WIDTH * EFF_CHANNELS + kw * EFF_CHANNELS + ch_ptr + 1;
end else begin : dws_cnv_st
   assign inc_start_pos = ch_ptr * KERNEL_WIDTH * KERNEL_HEIGHT + kh * KERNEL_WIDTH + kw + 1 ;
end
endgenerate

wire [$clog2(KERNEL_HEIGHT * KERNEL_WIDTH * EFF_CHANNELS) - 1: 0] adv_inc;
generate if (DWS == 0) begin
assign adv_inc = kh * KERNEL_WIDTH * EFF_CHANNELS + kw * EFF_CHANNELS + ch_ptr;
end else begin
assign adv_inc = ch_ptr * KERNEL_WIDTH * KERNEL_HEIGHT + kh * KERNEL_WIDTH + kw + 1;
end
endgenerate

genvar posvar;
wire [$clog2(BUFFER_SIZE+EFF_CHANNELS*(KERNEL_WIDTH+((KERNEL_HEIGHT-1)*IFMWidth)) + EFF_CHANNELS)-1 : 0] first_pos[MMV_OUT - 1:0];
generate if (MMV_IN == 1 || EFF_CHANNELS == 1) begin : mmvo_0_pos
   for(posvar = 0; posvar < MMV_OUT; posvar = posvar + 1) begin
   assign first_pos[posvar] =  starting_pos[posvar] + kw_tracker[posvar] * EFF_CHANNELS + kh_tracker[posvar]*(IFMWidth * EFF_CHANNELS) + ch_ptr ; 
   end
end else begin : mmvo_0_pos_mod
   for(posvar = 0; posvar < MMV_OUT; posvar = posvar + 1) begin
   assign first_pos[posvar] = starting_pos[posvar] + kw_tracker[posvar] + MMV_IN * mmvshift[posvar] + kh_tracker[posvar] * (IFMWidth * EFF_CHANNELS) + (ch_ptr * MMV_IN); 
   end
end
endgenerate

always @(posedge aclk)
    if(~aresetn) 
         input_counter <= 0;
    else if(input_counter == (IFMHeight * IFMWidth/MMV_IN * EFF_CHANNELS))
         input_counter <= 0;
    else if(s_axis_hs)
         input_counter <= input_counter + 1;
         
always @(posedge aclk)
    if(~aresetn | input_counter == IFMHeight * IFMWidth/MMV_IN * EFF_CHANNELS) 
         finish_rds <= 0;
    else if(buffer_empty & input_counter !=0)
         finish_rds <= 1;

always @(posedge aclk)
    if(~aresetn | (restart) | finish_rds ) 
         new_rd_cntr <= 0;
    else if(s_axis_hs & pending_rd_cntr == 0 & crtcl_rd_cntr[0] == 0 & buffer_full & !inc_pending_rd_gen & !inc_pending_rd_last)
         new_rd_cntr <= new_rd_cntr + 1;
    else if(start_new_row && (ofm_row_tracker[0] > PADDING_HEIGHT) & (new_rd_cntr > pending_rd_cntr + IFMWidth/MMV_IN * EFF_CHANNELS))
          new_rd_cntr <= new_rd_cntr - (pending_rd_cntr + IFMWidth/MMV_IN * EFF_CHANNELS) ;
    else if(start_new_row && (ofm_row_tracker[0] > PADDING_HEIGHT) & (new_rd_cntr <= pending_rd_cntr + IFMWidth/MMV_IN * EFF_CHANNELS))
          new_rd_cntr <= 0;
    else if(start_new_row && (ofm_row_tracker[0] == PADDING_HEIGHT) & (new_rd_cntr > pending_rd_cntr ))
          new_rd_cntr <= new_rd_cntr - pending_rd_cntr;
    else if(start_new_row && (ofm_row_tracker[0] == PADDING_HEIGHT) & (new_rd_cntr <= pending_rd_cntr))
          new_rd_cntr <= 0;
    
always @(posedge aclk)
    if(~aresetn | (restart) | finish_rds) 
        total_pending_rds <= BUFFER_SIZE/MMV_IN; //initial values
    else if(s_axis_hs & !inc_pending_rd_gen & !inc_pending_rd_last & !inc_pending_rd_stride)
        total_pending_rds <= total_pending_rds - 1; // whenever you get a new read
    else if( !(s_axis_hs) & inc_pending_rd_gen)  
        total_pending_rds <= total_pending_rds + inc_rd_amt;  
    else if (s_axis_hs & inc_pending_rd_gen)  
        total_pending_rds <= total_pending_rds + inc_rd_amt - 1;   
    else if ( !(s_axis_hs) & inc_pending_rd_last)
        total_pending_rds <=  total_pending_rds + inc_rd_amt_last;
    else if ( (s_axis_hs) & inc_pending_rd_last)
        total_pending_rds <=  total_pending_rds + inc_rd_amt_last - 1;        
    else if(!(s_axis_hs) & inc_pending_rd_stride)
        total_pending_rds <= total_pending_rds + IFMWidth/MMV_IN * EFF_CHANNELS;
    else if((s_axis_hs) & inc_pending_rd_stride)
        total_pending_rds <= total_pending_rds + IFMWidth/MMV_IN *EFF_CHANNELS - 1;
 
always @(posedge aclk)
    if(~aresetn | (restart) | finish_rds) 
        pending_rd_cntr <= BUFFER_SIZE/MMV_IN;
    else if (valid_ptr_vals && (ch_ptr == EFF_CHANNELS - 1 ) && (kw==KERNEL_WIDTH-1 && kh==KERNEL_HEIGHT-1) && (ofm_column_tracker[0] >= OFMWidth - MMV_OUT) && (ofm_row_tracker[0] >= PADDING_HEIGHT)) 
         pending_rd_cntr <= 0;
    else if(s_axis_hs & !inc_pending_rd_gen & !inc_pending_rd_last & pending_rd_cntr > 0 & crtcl_rd_cntr[0] == 0)
        pending_rd_cntr <= pending_rd_cntr - 1;
    else if( !(s_axis_hs) & inc_pending_rd_gen)  
        pending_rd_cntr <= pending_rd_cntr + inc_rd_amt; 
    else if ( !(s_axis_hs) & inc_pending_rd_last)
        pending_rd_cntr <=  pending_rd_cntr + inc_rd_amt_last; 
    else if (s_axis_hs & inc_pending_rd_gen & (crtcl_rd_cntr[0] != 0))
        pending_rd_cntr <= pending_rd_cntr + inc_rd_amt;
    else if (s_axis_hs & inc_pending_rd_last & (crtcl_rd_cntr[0] != 0))
        pending_rd_cntr <= pending_rd_cntr + inc_rd_amt_last;         
    else if (s_axis_hs & inc_pending_rd_gen & crtcl_rd_cntr[0] == 0)  
        pending_rd_cntr <= pending_rd_cntr + inc_rd_amt - 1;   
    else if (s_axis_hs & inc_pending_rd_last & crtcl_rd_cntr[0] == 0)  
        pending_rd_cntr <= pending_rd_cntr + inc_rd_amt_last - 1;
         
if( STRIDE_HT == 2 ) begin
  always @(posedge aclk) begin
    if(~aresetn | (restart) | finish_rds) begin
        crtcl_rd_cntr[0]<=0;
    end else if(!s_axis_hs && start_new_row && (ofm_row_tracker[0] > PADDING_HEIGHT)) begin
       if(new_rd_cntr < pending_rd_cntr +  IFMWidth/MMV_IN * EFF_CHANNELS)
          crtcl_rd_cntr[0] <= pending_rd_cntr +  IFMWidth/MMV_IN * EFF_CHANNELS - new_rd_cntr;
       else 
          crtcl_rd_cntr[0] <= 0;
    end else if(s_axis_hs && start_new_row && (ofm_row_tracker[0] > PADDING_HEIGHT)) begin
       if(new_rd_cntr < pending_rd_cntr +  IFMWidth/MMV_IN * EFF_CHANNELS)
          crtcl_rd_cntr[0] <= pending_rd_cntr +  IFMWidth/MMV_IN * EFF_CHANNELS - new_rd_cntr - 1;
       else 
          crtcl_rd_cntr[0] <= 0;
    end else if(!s_axis_hs && start_new_row && (ofm_row_tracker[0] == PADDING_HEIGHT)) begin
       if(new_rd_cntr < pending_rd_cntr)
          crtcl_rd_cntr[0] <= pending_rd_cntr- new_rd_cntr;
       else
          crtcl_rd_cntr[0] <=  0;        
    end else if(s_axis_hs && start_new_row && (ofm_row_tracker[0] == PADDING_HEIGHT)) begin
       if(new_rd_cntr < pending_rd_cntr)
          crtcl_rd_cntr[0] <= pending_rd_cntr- new_rd_cntr - 1;
       else
          crtcl_rd_cntr[0] <=  0; 
    end else if(s_axis_hs && crtcl_rd_cntr[0]>0) begin
        crtcl_rd_cntr[0] <= crtcl_rd_cntr[0] - 1;
    end
    end
end




//8
else begin
  always @(posedge aclk) begin : crtcl_cntr
    if(~aresetn | restart | finish_rds) begin
       crtcl_rd_cntr[0] <= 0;
    end else begin
      if ( start_new_row && !(s_axis_hs)) begin
          crtcl_rd_cntr[0] <= pending_rd_cntr;
      end else if (start_new_row && (s_axis_hs)) begin
          crtcl_rd_cntr[0] <= pending_rd_cntr - 1;
      end else if (crtcl_rd_cntr[0] > 0 && s_axis_hs) begin
          crtcl_rd_cntr[0] <= crtcl_rd_cntr[0] - 1;
      end            
    end
  end
end

always @(posedge aclk) begin : zeropad_reg_blk
   reg [MMV_OUT - 1: 0] i;
   if(~aresetn | restart | finish_rds)
     for( i = 0; i < MMV_OUT; i = i + 1) begin
       zeropad_reg[i] <= 0;
     end
   else if(valid_ptr_vals)
     for( i = 0; i < MMV_OUT;i = i + 1) begin
       zeropad_reg[i] <= (ofm_row_tracker[i] < PADDING_HEIGHT && kh < PADDING_HEIGHT) || (ofm_row_tracker[i] >= OFMHeight - PADDING_HEIGHT && kh >= KERNEL_HEIGHT - PADDING_HEIGHT) || (ofm_column_tracker[i] < PADDING_WIDTH && kw < PADDING_WIDTH) || (ofm_column_tracker[i] >= OFMWidth - PADDING_WIDTH && kw >= KERNEL_WIDTH - PADDING_WIDTH);
     end
   end
//8

always @(posedge aclk) begin : crtclrdcntr_reg_blk
   if(~aresetn | restart |finish_rds)
       crtcl_rd_cntr_del <= 0;
   else 
       crtcl_rd_cntr_del <= crtcl_rd_cntr[0];
end
   




always @(posedge aclk)
    if(~aresetn | finish_rds)
        q_valid <= 0;
    else if(enaB_q)
        q_valid <= r_valid & ~buffer_empty_i;
        
always @(posedge aclk)
    if(~aresetn | finish_rds)
        r_valid <= 0;
    else if(enaB_q | ~r_valid)
        r_valid <=  enaB;
        
always @(posedge aclk)
    if(~aresetn | finish_rds)
        enaB_reg <= 0;
    else 
        enaB_reg <=  enaB;        
//1
//assign m_axis_tvalid = buffer_full_i && !buffer_empty_i;
always @(posedge aclk) begin
if (~aresetn | finish_rds) begin
  buffer_empty_i <= 0;
  buffer_empty_ii <= 0;
  end else if (m_axis_hs) begin
    buffer_empty_i <= buffer_empty;
    buffer_empty_ii <= buffer_empty_i;
  end
  end

  
//2
always @(posedge aclk) begin
if (~aresetn | restart | finish_rds) begin
  buffer_empty <= 0;
  end
else if (start_new_row & ofm_row_tracker[0] == OFMHeight - 1 ) begin 
  buffer_empty <= 1;
end
end

// process to read data
//3
always @(*) begin : assign_pos
   if (~aresetn | restart | finish_rds) begin
     for (m = 0; m < MMV_OUT; m = m + 1) begin
        pos[m] = 0;
     end
   end else begin
      for (m = 0; m < MMV_OUT; m = m + 1) begin
         if (first_pos[m] >= BUFFER_SIZE ) 
            pos[m] = first_pos[m] - BUFFER_SIZE;
         else
            pos[m] = first_pos[m];
      end
   end
end


//4
always @(posedge aclk) begin
   if (~aresetn | restart | finish_rds) begin
      buffer_full_i <= 0;
   end else begin
      buffer_full_i <= buffer_full;
   end
end
  
//5
// process to write data
always @(posedge aclk) begin
   if (~aresetn | restart | finish_rds) begin
      counter <= 0;
      buffer_full <= 0;
      input_pixel <= 0;  
   end else begin      
      if(weA) begin
         if (counter < BUFFER_SIZE/MMV_IN - 1) begin
            counter <=counter + 1;
         end else begin
            counter <= 0;
            buffer_full <= 1;
            input_pixel <= input_pixel + 1;
         end 
         if (counter == ALLOW_WRITES  - 1) 
            start_write <= 1;      
      end
   end
end
//6
always @(posedge aclk) begin
   if (~aresetn | restart | finish_rds) begin
      ch_ptr <= 0;  
   end else begin
      if (inc_ch_ptr) begin
         if ((ch_ptr < EFF_CHANNELS - 1)) begin
            ch_ptr <= ch_ptr + 1;
         end else begin
            ch_ptr <= 0;
         end
      end
   end
end

always @(posedge aclk) begin : kh_kw_tracker
   integer s;
   if (~aresetn | restart |  finish_rds) begin
      for(s = 0; s < MMV_OUT; s = s + 1) begin
      kw_tracker[s] <= 0;
      kh_tracker[s] <= 0;
      if(s <= PADDING_WIDTH)
         mmv_sub_tracker[s] <= 0;
      else
         mmv_sub_tracker[s] <= (s-PADDING_WIDTH) * EFF_CHANNELS;
      end 
   end else if (inc_kw) begin
       for(s = 0; s < MMV_OUT; s = s + 1) begin
       if((kw != KERNEL_WIDTH - 1) && (kw_tracker[s] < KERNEL_WIDTH - 1) && (( ofm_column_tracker[s] >= PADDING_WIDTH) || (kw >= PADDING_WIDTH) ) && ((ofm_column_tracker[s] < (OFMWidth - PADDING_WIDTH)) || (kw< (KERNEL_WIDTH - PADDING_WIDTH - 1)) ) ) begin
          kw_tracker[s] <= kw_tracker[s] + 1;
          if (mmv_sub_tracker[s] < MMV_IN - 1) 
            mmv_sub_tracker[s] <= mmv_sub_tracker[s] + 1;
          else 
            mmv_sub_tracker[s] <= 0; 
       end else if (kw == KERNEL_WIDTH - 1) begin 
          kw_tracker[s] <= 0;
          mmv_sub_tracker[s] <= mmv_col_tracker_advance[s]; 
          if((kh != KERNEL_HEIGHT - 1) && (kh_tracker[s] < KERNEL_HEIGHT - 1) && (( ofm_row_tracker[s] >= PADDING_HEIGHT) || (kh >= PADDING_HEIGHT) ) && ((ofm_row_tracker[s] < (OFMHeight - PADDING_HEIGHT)) || (kh <(KERNEL_HEIGHT - PADDING_HEIGHT - 1)) ) ) 
            kh_tracker[s] <= kh_tracker[s] + 1;
          else if (kh == KERNEL_HEIGHT - 1)  
            kh_tracker[s] <= 0;
       end    
       end
   end
end

//7
always @(posedge aclk) begin : kh_kw
   integer s;
   if (~aresetn | restart | finish_rds) begin
      kw <= 0;
      kh <= 0; 
   end else
   if (inc_kw) begin
      if ((kw < KERNEL_WIDTH - 1)) begin
         kw <= kw + 1;           
      end else if (kw == KERNEL_WIDTH - 1) begin
         kw <= 0;        
         if (kh < (KERNEL_HEIGHT - 1) )begin
            kh <= kh + 1;
         end else begin
            kh <= 0;
         end
      end
   end
end

//8

always @(posedge aclk) begin : column_trackers
   //reg [$clog2(MMV_OUT) - 1:0] i;
   integer t;
   if(~aresetn | restart | finish_rds) begin
       for(t = 0; t < MMV_OUT; t = t + 1) begin
         ofm_column_tracker[t] <= t;
         ofm_row_tracker[t] <= 0;
         if(t <= PADDING_WIDTH)
            mmv_col_tracker[t] <= 0;
         else
            mmv_col_tracker[t] <= (t-PADDING_WIDTH) * EFF_CHANNELS;
       end
   end else begin
   if (valid_ptr_vals && (ch_ptr == EFF_CHANNELS - 1 ) && (kw==KERNEL_WIDTH-1 && kh==KERNEL_HEIGHT-1)) begin
   for(t = 0; t < MMV_OUT; t = t + 1) begin
       if(ofm_column_tracker[0] < (OFMWidth - MMV_OUT)) begin
          ofm_column_tracker[t] <= ofm_column_tracker[t] + MMV_OUT;
          mmv_col_tracker[t] <= mmv_col_tracker_advance[t];
       end else begin
          ofm_column_tracker[t] <= t;
          mmv_col_tracker[t] <= mmv_col_tracker_advance[t];
          if (ofm_row_tracker[t] < (OFMHeight - 1)) begin
             ofm_row_tracker[t] <= ofm_row_tracker[t] + 1;
          end else begin
             ofm_row_tracker[t] <= 0;
          end
       end
   end
   end
   end            
end

always @(posedge aclk) begin : kernel_row_trackers
   //reg [$clog2(MMV_OUT) - 1:0] i;
   integer t;
   if(~aresetn | restart | finish_rds) begin
      kernel_row_tracker <= 0;
   end else begin
   if (valid_ptr_vals && inc_start_pos == KERNEL_WIDTH*KERNEL_HEIGHT*EFF_CHANNELS - 2) begin
       if(ofm_column_tracker[0] >= OFMWidth - MMV_OUT) begin
          if (ofm_row_tracker[MMV_OUT - 1] < (OFMHeight - 1)) begin
                  if(ofm_row_tracker[0] < PADDING_HEIGHT)
                      kernel_row_tracker <= kernel_row_tracker + STRIDE_HT - PADDING_HEIGHT;
                  else if(kernel_row_tracker + STRIDE_HT > KERNEL_HEIGHT + STRIDE_HT - 1 - 1) 
                      kernel_row_tracker <= kernel_row_tracker + STRIDE_HT - KERNEL_HEIGHT - (STRIDE_HT - 1);
                  else if (kernel_row_tracker + STRIDE_HT <= KERNEL_HEIGHT + STRIDE_HT - 1 - 1) 
                      kernel_row_tracker <= kernel_row_tracker + STRIDE_HT;
          end else begin
             kernel_row_tracker <= 0;
          end
       end
   end
   end
               
end
//9
//DWS CAREFUL)
always @(posedge aclk) begin : col_tracker_adv
   integer i;
   if(~aresetn | restart | finish_rds) begin
      for(i = 0 ; i < MMV_OUT; i = i + 1) 
         if(i <= PADDING_WIDTH)
           mmv_col_tracker_advance[i] <= 0;
         else
           mmv_col_tracker_advance[i] <= (i-PADDING_WIDTH) * EFF_CHANNELS;
   end else begin
       if (valid_ptr_vals && adv_inc == KERNEL_WIDTH * KERNEL_HEIGHT * EFF_CHANNELS - 3) begin       
          if(ofm_column_tracker[0] < (OFMWidth - MMV_OUT))
             for (i = 0; i < MMV_OUT; i = i + 1) begin 
                if (STRIDE_WT == 1) begin
                   if(ofm_column_tracker[i] >= PADDING_WIDTH) begin
                      if (mmv_col_tracker_advance[i] + MMV_OUT >= MMV_IN * floor_O_BY_I && mmv_col_tracker_advance[i] + MMV_OUT < MMV_IN * (floor_O_BY_I + 1))
                        mmv_col_tracker_advance[i] <= mmv_col_tracker_advance[i] + MMV_OUT - MMV_IN * floor_O_BY_I;
                      else if (mmv_col_tracker_advance[i] + MMV_OUT >= MMV_IN * (floor_O_BY_I + 1))
                         mmv_col_tracker_advance[i] <= mmv_col_tracker_advance[i] + MMV_OUT - MMV_IN * (floor_O_BY_I + 1);
                   end else begin
                      if (mmv_col_tracker_advance[i] + MMV_OUT <= MMV_IN * floor_O_BY_I)
                         mmv_col_tracker_advance[i] <= mmv_col_tracker_advance[i] + MMV_OUT - PADDING_WIDTH;
                      else if (mmv_col_tracker_advance[i] + MMV_OUT > MMV_IN * floor_O_BY_I && mmv_col_tracker_advance[i] + MMV_OUT < MMV_IN * (floor_O_BY_I + 1))
                         mmv_col_tracker_advance[i] <= mmv_col_tracker_advance[i] + MMV_OUT - MMV_IN * floor_O_BY_I - PADDING_WIDTH;
                      else if (mmv_col_tracker_advance[i] + MMV_OUT >= MMV_IN * (floor_O_BY_I + 1))
                         mmv_col_tracker_advance[i] <= mmv_col_tracker_advance[i] + MMV_OUT - MMV_IN * (floor_O_BY_I + 1) - PADDING_WIDTH;                  
                   end             
                end else  begin
                   if(ofm_column_tracker[0] < PADDING_WIDTH) begin
                       if(mmv_col_tracker[i] + STRIDE_WT - PADDING_WIDTH < MMV_IN)
                          mmv_col_tracker_advance[i] <= mmv_col_tracker[i] + STRIDE_WT - PADDING_WIDTH;
                       else
                          mmv_col_tracker_advance[i] <= mmv_col_tracker[i] + STRIDE_WT - PADDING_WIDTH- MMV_IN;                  
                   end else begin                
                       if(mmv_col_tracker[i] + STRIDE_WT < MMV_IN)
                          mmv_col_tracker_advance[i] <= mmv_col_tracker[i] + STRIDE_WT;
                       else
                          mmv_col_tracker_advance[i] <= mmv_col_tracker[i] + STRIDE_WT - MMV_IN;
                   end 
                end 
             end
          else 
            for(i = 0 ; i < MMV_OUT; i = i + 1) 
              if(i <= PADDING_WIDTH)
                mmv_col_tracker_advance[i] <= 0;
              else
                mmv_col_tracker_advance[i] <= (i-PADDING_WIDTH) * EFF_CHANNELS;
      end
   end
end          



//10
always @(posedge aclk) begin : starting_pos_blk
integer r;

   //reg [$clog2(MMV_OUT) - 1:0] i;
   if (~aresetn | restart | finish_rds) begin
      for (r = 0; r < MMV_OUT; r = r + 1) begin
         starting_pos[r] <= 0;
      end
   end else if (valid_ptr_vals || (!buffer_full & counter == ALLOW_WRITES - 2)) begin // early start
      for (r = 0; r < MMV_OUT; r = r + 1) begin
         if(starting_pos_i[r] >= BUFFER_SIZE) 
            starting_pos[r] <= starting_pos_i[r] - BUFFER_SIZE;
         else 
            starting_pos[r] <= starting_pos_i[r];
      end
   end
end



//11
always @(posedge aclk) begin : start_pos_i_blk
   integer i;
   if (~aresetn |restart | finish_rds) begin
         for(i = 0 ; i < MMV_OUT; i = i + 1) 
            if(i <= PADDING_WIDTH)
            starting_pos_i[i] <= 0;
            else
            starting_pos_i[i] <= (i-PADDING_WIDTH) * EFF_CHANNELS * STRIDE_WT ;
  end else begin
      if (valid_ptr_vals && inc_start_pos==KERNEL_WIDTH*KERNEL_HEIGHT*EFF_CHANNELS - 1) begin // if the column will be updated in the next cycle
         if ((ofm_column_tracker[0] < (OFMWidth - MMV_OUT ))) begin //should not increment by one if it's less than padding_width or greater than End-width - padding-width
            if (MMV_IN == 1 || EFF_CHANNELS == 1) begin  
               for(i = 0; i < MMV_OUT; i= i + 1) begin
                  if(ofm_column_tracker[i] < PADDING_WIDTH)
                     starting_pos_i[i] = starting_pos[i] + (STRIDE_WT*MMV_OUT) * EFF_CHANNELS - PADDING_WIDTH ; 
                  else
                     starting_pos_i[i] = starting_pos[i] + (STRIDE_WT*MMV_OUT) * EFF_CHANNELS; 
                end
            end else begin 
               for(i = 0; i < MMV_OUT; i= i + 1) begin
                  if (ofm_column_tracker[i] < PADDING_WIDTH) begin
                  if (mmv_col_tracker[i] + MMV_OUT * STRIDE_WT > MMV_IN * floor_O_BY_I && mmv_col_tracker[i] + MMV_OUT * STRIDE_WT < MMV_IN * (floor_O_BY_I + 1)) begin
                     starting_pos_i[i] <= starting_pos[i] + MMV_OUT*STRIDE_WT + (MMV_IN * floor_O_BY_I) - PADDING_WIDTH;
                  end else if (mmv_col_tracker[i] + MMV_OUT * STRIDE_WT >= MMV_IN * (floor_O_BY_I + 1)) begin
                     starting_pos_i[i] <= starting_pos[i] + MMV_OUT*STRIDE_WT + MMV_IN * (floor_O_BY_I + 1) - PADDING_WIDTH; 
                  end
                  end else begin
                  if (mmv_col_tracker[i] + MMV_OUT * STRIDE_WT > MMV_IN * floor_O_BY_I && mmv_col_tracker[i] + MMV_OUT * STRIDE_WT < MMV_IN * (floor_O_BY_I + 1)) begin
                     starting_pos_i[i] <= starting_pos[i] + MMV_OUT*STRIDE_WT + (MMV_IN * floor_O_BY_I);
                  end else if (mmv_col_tracker[i] + MMV_OUT * STRIDE_WT >= MMV_IN * (floor_O_BY_I + 1)) begin
                     starting_pos_i[i] <= starting_pos[i] + MMV_OUT*STRIDE_WT + MMV_IN * (floor_O_BY_I + 1); 
                  end                  
                  end
                  
               end
            end                                                           
         end else if (ofm_column_tracker[0] + MMV_OUT == OFMWidth) begin// if switching to a new row
               for(i = 0; i < MMV_OUT; i = i + 1) begin
                   if(i < PADDING_WIDTH)
                       starting_pos_i[i] <= (kernel_row_tracker) * IFMWidth * EFF_CHANNELS + i * STRIDE_WT * EFF_CHANNELS;
                   else 
                       starting_pos_i[i] <= (kernel_row_tracker) * IFMWidth * EFF_CHANNELS + (i-PADDING_WIDTH) * EFF_CHANNELS *STRIDE_WT ;                        
               end
         end
      end
   end
end
//12
always @(posedge aclk) begin : mmvshift_blk
   integer n;
   if (~aresetn | restart | finish_rds) begin 
      for(n = 0; n < MMV_OUT; n = n + 1) begin 
         mmvshift[n] <= 0;
      end
   end else begin
      for(n = 0; n < MMV_OUT; n = n + 1) begin 
         if (mmv_sub_tracker[n]==MMV_IN-1 && (( ofm_column_tracker[n] >= PADDING_WIDTH) || (kw >= PADDING_WIDTH))  && (ofm_column_tracker[n] < OFMWidth - PADDING_WIDTH || (kw< (KERNEL_WIDTH - PADDING_WIDTH - 1))) && (ch_ptr == EFF_CHANNELS - 1 || DWS == 1) && EFF_CHANNELS > 1 && (kw != KERNEL_WIDTH - 1 && valid_ptr_vals)) 
           mmvshift[n] <= 1;
         else if (kw == KERNEL_WIDTH - 1 && (ch_ptr == EFF_CHANNELS - 1 || DWS == 1) && valid_ptr_vals)
           mmvshift[n] <= 0;
      end
   end
end


function integer log2;
input integer value;
reg [31:0] shifted;
integer res;
begin 
  if (value < 2)  
    log2 = value;
   else begin 
      shifted = value-1;  
      for (res=0; shifted>0; res=res+1)   
         shifted = shifted>>1;  
         log2 = res; 
      end
   end
endfunction
endmodule
