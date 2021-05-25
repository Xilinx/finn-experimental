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

module inputbuf
#(
    parameter WIDTH = 16,
    parameter DEPTH = 32,
    parameter NFOLDS = 8,
    parameter RAM_STYLE = "auto"
)
(
    input aclk,
    input aresetn,

    input s_axis_tvalid,
    input [WIDTH-1:0] s_axis_tdata,
    output s_axis_tready,

    output m_axis_tvalid,
    output [WIDTH-1:0] m_axis_tdata,
    input m_axis_tready
);

localparam DEPTH_LOG = $clog2(DEPTH);
localparam NFOLDS_LOG = $clog2(NFOLDS);

//store and repeatedly output a packet of DEPTH words
//for the first fold we read and cut through and simultaneously write to a RAM
//the subsequent times we read from the RAM
reg [NFOLDS_LOG-1:0] fold_count;
reg [DEPTH_LOG-1:0] ram_addr;
wire ram_we, ram_en, ram_enq;
reg r_valid, q_valid;
wire s_axis_tvalid_int;

wire reset_ram_addr, reset_fold_count;
assign reset_ram_addr = ram_en & (ram_addr == (DEPTH-1));
assign reset_fold_count = reset_ram_addr & (fold_count == (NFOLDS-1));


//we write the ram during the first fold when valid data on input
assign ram_we = (fold_count == 0) & s_axis_tvalid;
//enable the RAM read/write when
//a read is currently pending and accepted
//a read is not pending and if on the first fold we have data on s_axis
assign s_axis_tvalid_int = (fold_count != 0) | s_axis_tvalid;
assign ram_en = s_axis_tvalid_int & (ram_enq | ~r_valid);
//enable the Q register when no backpressure on output
assign ram_enq = m_axis_tready | ~q_valid;

assign s_axis_tready = ram_en & (fold_count == 0);
assign m_axis_tvalid = q_valid;

always @(posedge aclk)
    if(~aresetn | reset_fold_count)
        fold_count <= 0;
    else if(reset_ram_addr)
        fold_count <= fold_count + 1;

always @(posedge aclk)
    if(~aresetn | reset_ram_addr)
        ram_addr <= 0;
    else if(ram_en)
        ram_addr <= ram_addr + 1;

always @(posedge aclk)
    if(~aresetn)
        q_valid <= 0;
    else if(ram_enq)
        q_valid <= r_valid;

always @(posedge aclk)
    if(~aresetn)
        r_valid <= 0;
    else if(ram_enq | ~r_valid)
        r_valid <= ram_en;

ram_wf
#(
    .DWIDTH(WIDTH),
    .AWIDTH(DEPTH_LOG),
    .RAM_STYLE(RAM_STYLE)
)
activation_storage
(
	.clk(aclk),

	.en(ram_en),
	.enq(ram_enq),
	.we(ram_we),
	.addr(ram_addr),
	.wdata(s_axis_tdata),
	.rdq(m_axis_tdata)
);

endmodule
