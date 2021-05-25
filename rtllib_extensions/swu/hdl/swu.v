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

module swu #(
    parameter SIMD = 1,
    parameter STRIDE = 1,
    parameter IFMChannels = 2,
    parameter KERNEL_HEIGHT = 3,
    parameter KERNEL_WIDTH = 3,
    parameter RAM_STYLE = "auto",

    parameter IFMWidth = 8,
    parameter IFMHeight = 8,
    parameter PADDING_WIDTH = 0,
    parameter PADDING_HEIGHT =0,
    parameter OFMWidth = 6,
    parameter OFMHeight = 6,

    //depths per stream
    parameter PRECISION = 8,
    parameter MMV_IN = 2,
    parameter MMV_OUT = 1
)
(
    input aclk,
    input aresetn,

    input [MMV_IN * SIMD * PRECISION - 1 : 0] s_axis_tdata,
    input s_axis_tvalid,
    output s_axis_tready,

    output [MMV_OUT * SIMD * PRECISION - 1 : 0] m_axis_tdata,
    input m_axis_tready,
    output m_axis_tvalid
);

localparam BUFFER_DEPTH = KERNEL_HEIGHT * IFMWidth * (IFMChannels/SIMD) + KERNEL_WIDTH;
localparam EFF_CHANNELS = IFMChannels/SIMD;
localparam SIZEA = BUFFER_DEPTH/MMV_IN;
localparam SIZEB = BUFFER_DEPTH;
localparam ADDRWIDTHA = $clog2(SIZEA);
localparam ADDRWIDTHB = $clog2(SIZEB);
localparam WIDTHA = MMV_IN * SIMD * PRECISION ;
localparam WIDTHB = SIMD * PRECISION;

wire s_axis_hs;
wire m_axis_hs;

assign m_axis_hs = m_axis_tready & m_axis_tvalid;
assign s_axis_hs = s_axis_tready & s_axis_tvalid;

wire [$clog2(BUFFER_DEPTH/MMV_IN)-1:0] wr_addr;
wire [$clog2(BUFFER_DEPTH)-1:0] rd_addr[MMV_OUT-1:0];

wire [MMV_OUT-1:0] rd_done;
wire [MMV_OUT-1:0] rd_en;
wire [MMV_OUT-1:0] rd_enq;
wire buffer_full;

wr_control #(
    .NPIXELS(IFMWidth * IFMHeight),
    .WORDS_PER_PX(EFF_CHANNELS),
    .MMV_IN(MMV_IN),
    .BUFFER_DEPTH(SIZEA)
)
wrctrl
(
    .aclk(aclk),
    .aresetn(aresetn),

    .ready(s_axis_tready),
    .handshake(s_axis_hs),
    .restart(&rd_done),
    .full(buffer_full),

    .addr(wr_addr)
);

genvar i;
generate for(i=0; i<MMV_OUT; i=i+1) begin: mmv_output
    rd_control #(
        .NPIXELS(IFMWidth * IFMHeight),
        .WORDS_PER_PX(EFF_CHANNELS),
        .MMV_IN(MMV_IN),
        .IFMWidth(IFMWidth),
        .STRIDE(STRIDE),
        .KERNEL_HEIGHT(KERNEL_HEIGHT),
        .KERNEL_WIDTH(KERNEL_WIDTH),
        .OFMWidth(OFMWidth),
        .OFMHeight(OFMHeight),
        .PADDING_WIDTH(PADDING_WIDTH),
        .PADDING_HEIGHT(PADDING_HEIGHT),
        .BUFFER_DEPTH(SIZEB)
    )
    rdctrl
    (
        .aclk(aclk),
        .aresetn(aresetn),

        .wr_handshake(s_axis_hs),
        .wr_addr(wr_addr),

        .ready(m_axis_tready),
        .handshake(m_axis_hs),
        .done(rd_done[i]),
        .full(buffer_full),

        .addr(rd_addr[i]),
        .en(rd_en[i]),
        .enq(rd_enq[i]),
        .valid(m_axis_tvalid)
    );

    //buffer instance
    asymmetric_ram  #(
        .SIZEA(SIZEA),
        .SIZEB(SIZEB),
        .WIDTHA(WIDTHA),
        .WIDTHB(WIDTHB),
        .ADDRWIDTHA(ADDRWIDTHA),
        .ADDRWIDTHB(ADDRWIDTHB),
        .RAM_STYLE(RAM_STYLE)
    )
    ram
    (
        .clkA(aclk),
        .enaA(s_axis_hs),
        .weA(weA),
        .addrA(wr_addr),
        .diA(s_axis_tdata),

        .clkB(aclk),
        .enaB(rd_en[i]),
        .enaB_q(rd_enq[i]),
        .addrB(rd_addr[i]),
        .doB(m_axis_tdata[(i+1)*SIMD*PRECISION-1:0])
    );
end
endgenerate

endmodule
