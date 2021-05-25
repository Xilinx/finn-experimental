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

module tb;

parameter WIDTH = 16;
parameter DEPTH = 32;
parameter NFOLDS = 8;

reg aclk, aresetn;

reg s_axis_tvalid, m_axis_tready;
wire m_axis_tvalid, s_axis_tready;
reg [WIDTH-1:0] s_axis_tdata;
wire [WIDTH-1:0] m_axis_tdata;
integer i;
integer start_tvalid_tready_test = 0;

initial begin
    aclk = 0;
    forever #5 aclk = ~aclk;
end

initial begin
    aresetn = 0;
    s_axis_tvalid = 0;
    #1000
    @(negedge aclk);
    aresetn = 1;
    #1000
    @(negedge aclk);

    //push two packet through at full speed
    s_axis_tvalid = 1;
    m_axis_tready = 1;
    s_axis_tdata = 0;
    for(i=0; i<DEPTH; i=i+1) begin
        @(negedge aclk);
        while(s_axis_tvalid == 1 & s_axis_tready == 0)
            @(negedge aclk);
        s_axis_tdata = s_axis_tdata + 1;
    end
    for(i=0; i<DEPTH; i=i+1) begin
        @(negedge aclk);
        while(s_axis_tvalid == 1 & s_axis_tready == 0)
            @(negedge aclk);
        s_axis_tdata = s_axis_tdata + 1;
    end

    //push two packets through with random intervals between tvalid/tready
    s_axis_tdata = 100;
    start_tvalid_tready_test = 1;
    for(i=0; i<DEPTH; i=i+1) begin
        repeat($random()%3) begin
            s_axis_tvalid = 0;
            @(negedge aclk);
        end
        s_axis_tvalid = 1;
        @(negedge aclk);
        while(s_axis_tvalid == 1 & s_axis_tready == 0)
            @(negedge aclk);
        s_axis_tdata = s_axis_tdata + 1;
    end
    for(i=0; i<DEPTH; i=i+1) begin
        repeat($random()%5) begin
            s_axis_tvalid = 0;
            @(negedge aclk);
        end
        s_axis_tvalid = 1;
        @(negedge aclk);
        while(s_axis_tvalid == 1 & s_axis_tready == 0)
            @(negedge aclk);
        s_axis_tdata = s_axis_tdata + 1;
    end
    s_axis_tvalid = 0;
end

initial begin
    m_axis_tready = 1;
    @(start_tvalid_tready_test == 1);
    forever begin
         @(posedge aclk);
         #1
         m_axis_tready = ~m_axis_tready;
    end
end

inputbuf
#(
    .WIDTH(WIDTH),
    .DEPTH(DEPTH),
    .NFOLDS(NFOLDS)
)
dut
(
    .aclk(aclk),
    .aresetn(aresetn),

    .s_axis_tvalid(s_axis_tvalid),
    .s_axis_tdata(s_axis_tdata),
    .s_axis_tready(s_axis_tready),

    .m_axis_tvalid(m_axis_tvalid),
    .m_axis_tdata(m_axis_tdata),
    .m_axis_tready(m_axis_tready)
);

endmodule
