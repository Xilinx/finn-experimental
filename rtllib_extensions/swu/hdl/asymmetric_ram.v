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

module asymmetric_ram
#(
parameter WIDTHB = 4,
parameter SIZEB = 1024,
parameter ADDRWIDTHB = 10,
parameter WIDTHA = 16,
parameter SIZEA = 256,
parameter ADDRWIDTHA = 8,
parameter RAM_STYLE = "auto"
)
(
input clkA, 
input clkB, 
input weA, 
input enaA, 
input enaB,
input enaB_q, 
 
input [ADDRWIDTHA - 1 : 0] addrA, 
input [ADDRWIDTHB - 1 : 0] addrB, 
input [WIDTHA - 1 : 0] diA, 
output reg [WIDTHB-1:0] doB    );
    
`define min(a,b) {(a) < (b) ? (a) : (b)}
`define max(a,b) {(a) > (b) ? (a) : (b)}
localparam maxSIZE = `max(SIZEA, SIZEB);
localparam maxWIDTH = `max(WIDTHA, WIDTHB);
localparam minWIDTH = `min(WIDTHA, WIDTHB);
localparam RATIO = maxWIDTH / minWIDTH;
localparam log2RATIO = log2(RATIO);

(* ram_style = RAM_STYLE *) reg [minWIDTH-1:0] RAM [0:maxSIZE-1];
reg [WIDTHB-1:0] readB;

always @(posedge clkB) begin 
  if (enaB) begin  
     readB <= RAM[addrB]; 
  end 
  if (enaB_q) begin
     doB <= readB;
  end
end

always @(posedge clkA) begin : ramwrite
   integer i; 
   reg [$clog2(RATIO)-1:0] lsbaddr; 
   for (i=0; i< RATIO; i= i+ 1) begin : write1  
      lsbaddr = i;  
      if (enaA) begin  
         if (weA)    
            RAM[addrA * RATIO  + lsbaddr] <= diA[(i+1)*minWIDTH-1 -: minWIDTH];  
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

