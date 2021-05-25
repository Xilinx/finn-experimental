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

module wr_control #(
    parameter NPIXELS = 1024,
    parameter PX_PER_WORD = 1,
    parameter MMV_IN = 2,
    parameter BUFFER_DEPTH = 20
)
(
    input aclk,
    input aresetn,

    output ready,
    input handshake,
    input restart,
    output reg full,

    output reg [$clog2(BUFFER_DEPTH/MMV_IN)-1:0] addr
);

//buffer write logic
reg [$clog2(NPIXELS) - 1 : 0] input_pixel= 0;
reg [$clog2(BUFFER_DEPTH/MMV_IN) - 1 : 0] pending_rd_cntr = BUFFER_DEPTH/MMV_IN;

assign ready = !full || (pending_rd_cntr > 0);
assign weA = handshake & ( (input_pixel * BUFFER_DEPTH + addr) < (NPIXELS * PX_PER_WORD));

always @(posedge aclk) begin
    if (~aresetn) begin
        addr <= 0;
        full <= 0;
        input_pixel <= 0;
    end else if(restart) begin
        addr <= 0;
        full <= 0;
        input_pixel <= 0;   
    end else begin      
        if(weA) begin
            if (addr < BUFFER_DEPTH/MMV_IN - 1) begin
                addr <=addr + 1;
            end else begin
                addr <= 0;
                full <= 1;
                input_pixel <= input_pixel + 1;
            end
            
        end
    end
end

endmodule
