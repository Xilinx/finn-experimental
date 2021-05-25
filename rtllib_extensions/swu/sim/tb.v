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

module tb();

parameter MMV_IN = 3;
parameter MMV_OUT = 1;
parameter SIMD = 1;
parameter IP_PRECISION = 8;

parameter STRIDE = 2;
parameter IFMChannels = 2;
parameter KERNEL_HEIGHT = 3;
parameter KERNEL_WIDTH = 3;
parameter RAM_STYLE = "auto";
parameter IFMWidth = 6;
parameter IFMHeight = 6;
parameter PADDING_WIDTH = 0;
parameter PADDING_HEIGHT =0;
parameter OFMWidth = 2;
parameter OFMHeight = 2;
parameter BUFFER_SIZE = 36;
parameter OFMDIM_MOD_MMV = 1;
	
reg clk;
reg resetn;
reg [MMV_IN * SIMD * IP_PRECISION - 1 : 0] ip_axis_tdata;
reg ip_axis_tvalid;
wire ip_axis_tready;
wire [SIMD * IP_PRECISION - 1 : 0] op_axis_tdata;
reg op_axis_tready;
wire op_axis_tvalid;

swu #(
    .MMV_IN(MMV_IN),
    .MMV_OUT(MMV_OUT),
    .SIMD(SIMD),
    .IP_PRECISION(IP_PRECISION),
    .STRIDE(STRIDE),
    .IFMChannels(IFMChannels),
    .KERNEL_HEIGHT(KERNEL_HEIGHT),
    .KERNEL_WIDTH(KERNEL_WIDTH), 
    .RAM_STYLE(RAM_STYLE),
    .IFMWidth(IFMWidth),
    .IFMHeight(IFMHeight),
    .PADDING_WIDTH(PADDING_WIDTH),
    .PADDING_HEIGHT(PADDING_HEIGHT), 
    .OFMWidth(OFMWidth), 
    .OFMHeight(OFMHeight),
    .BUFFER_SIZE(BUFFER_SIZE),
    .OFMDIM_MOD_MMV(OFMDIM_MOD_MMV)
) uut (
    .aclk(clk),
    .aresetn(resetn),
    .s_axis_tdata(ip_axis_tdata),
    .s_axis_tvalid(ip_axis_tvalid),
    .s_axis_tready(ip_axis_tready),
    .m_axis_tdata(op_axis_tdata),
    .m_axis_tready(op_axis_tready),
    .m_axis_tvalid(op_axis_tvalid)
);

integer i=0;
initial begin
    clk = 0;
    forever #10 clk = ~clk;
end

initial begin
    op_axis_tready = 0;
    #960;
    op_axis_tready = 1;
    #80;
    op_axis_tready = 0;
    #60;
    op_axis_tready = 1;
    #210;
    op_axis_tready = 0;
    #50;
    op_axis_tready = 1;
end

initial begin
    resetn = 0;
    ip_axis_tvalid = 0;
    ip_axis_tdata = 24'h030201;
    //op_axis_tready = 0;
    #500;
    resetn = 1;
    //op_axis_tready = 1;

    @(negedge clk);
    #300;
    //ip_axis_tvalid = 1;

    while(i<6400) begin
        if (ip_axis_tready == 1) begin
            i = i + 1;
            if(i%2 == 0)begin
                ip_axis_tdata = ip_axis_tdata + 24'h030303;
            end
            if(i % 16 == 0) begin
                ip_axis_tvalid = 1;
                //end else begin
                //   ip_axis_tvalid = 0;
            end
        end

        #20;  
    end
end
endmodule
