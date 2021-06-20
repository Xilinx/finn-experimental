`timescale 1ns / 1ps

module asymmetrc_ram
#(
parameter WIDTHB = 4,
parameter SIZEB = 1024,
parameter ADDRWIDTHB = 10,
parameter WIDTHA = 16,
parameter SIZEA = 256,
parameter ADDRWIDTHA = 8,
parameter RAM_STYLE = "auto")


(
input clkA, 
input clkB, 
input weA, 
input enaA, 
input enaB,
input enaB_q, 
input zeropad,
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
     if (zeropad)
       doB <= 0;
     else
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

