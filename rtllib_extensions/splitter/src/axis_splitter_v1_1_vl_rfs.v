// -- (c) Copyright 2011-2012 Xilinx, Inc. All rights reserved.
// --
// -- This file contains confidential and proprietary information
// -- of Xilinx, Inc. and is protected under U.S. and 
// -- international copyright and other intellectual property
// -- laws.
// --
// -- DISCLAIMER
// -- This disclaimer is not a license and does not grant any
// -- rights to the materials distributed herewith. Except as
// -- otherwise provided in a valid license issued to you by
// -- Xilinx, and to the maximum extent permitted by applicable
// -- law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
// -- WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
// -- AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
// -- BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
// -- INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and
// -- (2) Xilinx shall not be liable (whether in contract or tort,
// -- including negligence, or under any other theory of
// -- liability) for any loss or damage of any kind or nature
// -- related to, arising under or in connection with these
// -- materials, including for any direct, or any indirect,
// -- special, incidental, or consequential loss or damage
// -- (including loss of data, profits, goodwill, or any type of
// -- loss or damage suffered as a result of any action brought
// -- by a third party) even if such damage or loss was
// -- reasonably foreseeable or Xilinx had been advised of the
// -- possibility of the same.
// --
// -- CRITICAL APPLICATIONS
// -- Xilinx products are not designed or intended to be fail-
// -- safe, or for use in any application requiring fail-safe
// -- performance, such as life-support or safety devices or
// -- systems, Class III medical devices, nuclear facilities,
// -- applications related to the deployment of airbags, or any
// -- other applications that could lead to death, personal
// -- injury, or severe property or environmental damage
// -- (individually and collectively, "Critical
// -- Applications"). Customer assumes the sole risk and
// -- liability of any use of Xilinx products in Critical
// -- Applications, subject only to applicable laws and
// -- regulations governing limitations on product liability.
// --
// -- THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
// -- PART OF THIS FILE AT ALL TIMES.
//-----------------------------------------------------------------------------
//
// AXIS Broadcaster
//   Generic single-channel AXIS to multiple channel AXIS.
//
// Verilog-standard:  Verilog 2001
//--------------------------------------------------------------------------
//
//--------------------------------------------------------------------------

`timescale 1ps/1ps
`default_nettype none

module axis_split_core #(
///////////////////////////////////////////////////////////////////////////////
// Parameter Definitions
///////////////////////////////////////////////////////////////////////////////
   parameter         C_FAMILY           = "rtl",
   parameter integer C_AXIS_TDATA_WIDTH = 80,
   parameter integer S_AXIS_TDATA_WIDTH_PAD = 80,
   parameter integer C_AXIS_TID_WIDTH   = 1,
   parameter integer C_AXIS_TDEST_WIDTH = 1,
   parameter integer C_AXIS_TUSER_WIDTH = 1,
   parameter [31:0]  C_AXIS_SIGNAL_SET  = 03'hFF,
   // C_AXIS_SIGNAL_SET: each bit if enabled specifies which axis optional signals are present
   //   [0] => TREADY present
   //   [1] => TDATA present
   //   [2] => TSTRB present, TDATA must be present
   //   [3] => TKEEP present, TDATA must be present
   //   [4] => TLAST present
   //   [5] => TID present
   //   [6] => TDEST present
   //   [7] => TUSER present
   parameter integer C_NUM_MI_SLOTS     = 10,
   parameter integer M_AXIS_TDATA_WIDTH_PAD = 8


) (
///////////////////////////////////////////////////////////////////////////////
// Port Declarations
///////////////////////////////////////////////////////////////////////////////
   // System Signals
   input wire aclk,
   input wire aresetn,
   input wire aclken,

   // Slave side
   input  wire                            s_axis_tvalid,
   output wire                            s_axis_tready,
   input  wire [S_AXIS_TDATA_WIDTH_PAD-1:0]   s_axis_tdata,
   /*input  wire [C_AXIS_TDATA_WIDTH/8-1:0] s_axis_tstrb,
   input  wire [C_AXIS_TDATA_WIDTH/8-1:0] s_axis_tkeep,
   input  wire                            s_axis_tlast,
   input  wire [C_AXIS_TID_WIDTH-1:0]     s_axis_tid,
   input  wire [C_AXIS_TDEST_WIDTH-1:0]   s_axis_tdest,
   input  wire [C_AXIS_TUSER_WIDTH-1:0]   s_axis_tuser,
*/
   // Master side
   output wire          m_axis_00_tvalid,
   input  wire           m_axis_00_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_00_tdata,

   output wire          m_axis_01_tvalid,
   input  wire           m_axis_01_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_01_tdata,

   output wire          m_axis_02_tvalid,
   input  wire           m_axis_02_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_02_tdata,

   output wire          m_axis_03_tvalid,
   input  wire           m_axis_03_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_03_tdata,
   
   output wire          m_axis_04_tvalid,
   input  wire           m_axis_04_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_04_tdata,

   output wire          m_axis_05_tvalid,
   input  wire           m_axis_05_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_05_tdata,

   output wire          m_axis_06_tvalid,
   input  wire           m_axis_06_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_06_tdata,

   output wire          m_axis_07_tvalid,
   input  wire           m_axis_07_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_07_tdata,

   output wire          m_axis_08_tvalid,
   input  wire           m_axis_08_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_08_tdata,

   output wire          m_axis_09_tvalid,
   input  wire           m_axis_09_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_09_tdata,

   output wire          m_axis_10_tvalid,
   input  wire           m_axis_10_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_10_tdata,

   output wire          m_axis_11_tvalid,
   input  wire           m_axis_11_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_11_tdata,

   output wire          m_axis_12_tvalid,
   input  wire           m_axis_12_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_12_tdata,

   output wire          m_axis_13_tvalid,
   input  wire           m_axis_13_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_13_tdata,

   output wire          m_axis_14_tvalid,
   input  wire           m_axis_14_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_14_tdata,

   output wire          m_axis_15_tvalid,
   input  wire           m_axis_15_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_15_tdata,
   
   output wire  m_axis_16_tvalid,
   input wire m_axis_16_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]   m_axis_16_tdata,

   output wire  m_axis_17_tvalid,
   input wire m_axis_17_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_17_tdata,

   output wire  m_axis_18_tvalid,
   input wire m_axis_18_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_18_tdata,

   output wire  m_axis_19_tvalid,
   input wire m_axis_19_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_19_tdata,

   output wire  m_axis_20_tvalid,
   input wire m_axis_20_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_20_tdata,

   output wire  m_axis_21_tvalid,
   input wire m_axis_21_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_21_tdata,

   output wire  m_axis_22_tvalid,
   input wire m_axis_22_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_22_tdata,

   output wire  m_axis_23_tvalid,
   input wire m_axis_23_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_23_tdata,

   output wire  m_axis_24_tvalid,
   input wire m_axis_24_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_24_tdata,

   output wire  m_axis_25_tvalid,
   input wire m_axis_25_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_25_tdata,

   output wire  m_axis_26_tvalid,
   input wire m_axis_26_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_26_tdata,

   output wire  m_axis_27_tvalid,
   input wire m_axis_27_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_27_tdata,

   output wire  m_axis_28_tvalid,
   input wire m_axis_28_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]   m_axis_28_tdata,

   output wire  m_axis_29_tvalid,
   input wire m_axis_29_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_29_tdata,

   output wire  m_axis_30_tvalid,
   input wire m_axis_30_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_30_tdata,

   output wire  m_axis_31_tvalid,
   input wire m_axis_31_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_31_tdata,

   output wire  m_axis_32_tvalid,
   input wire m_axis_32_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_32_tdata,

   output wire  m_axis_33_tvalid,
   input wire m_axis_33_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_33_tdata,

   output wire  m_axis_34_tvalid,
   input wire m_axis_34_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_34_tdata,

   output wire  m_axis_35_tvalid,
   input wire m_axis_35_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_35_tdata,

   output wire  m_axis_36_tvalid,
   input wire m_axis_36_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]   m_axis_36_tdata,

   output wire  m_axis_37_tvalid,
   input wire m_axis_37_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_37_tdata,

   output wire  m_axis_38_tvalid,
   input wire m_axis_38_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_38_tdata,

   output wire  m_axis_39_tvalid,
   input wire m_axis_39_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_39_tdata,

   output wire  m_axis_40_tvalid,
   input wire m_axis_40_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_40_tdata,

   output wire  m_axis_41_tvalid,
   input wire m_axis_41_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_41_tdata,

   output wire  m_axis_42_tvalid,
   input wire m_axis_42_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_42_tdata,

   output wire  m_axis_43_tvalid,
   input wire m_axis_43_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_43_tdata,

   output wire  m_axis_44_tvalid,
   input wire m_axis_44_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_44_tdata,

   output wire  m_axis_45_tvalid,
   input wire m_axis_45_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_45_tdata,

   output wire  m_axis_46_tvalid,
   input wire m_axis_46_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]   m_axis_46_tdata,

   output wire  m_axis_47_tvalid,
   input wire m_axis_47_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_47_tdata,

   output wire  m_axis_48_tvalid,
   input wire m_axis_48_tready,
   output wire[(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_48_tdata,

   output wire  m_axis_49_tvalid,
   input wire m_axis_49_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_49_tdata,

   output wire  m_axis_50_tvalid,
   input wire m_axis_50_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_50_tdata,

   output wire  m_axis_51_tvalid,
   input wire m_axis_51_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_51_tdata,

   output wire  m_axis_52_tvalid,
   input wire m_axis_52_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]   m_axis_52_tdata,

   output wire  m_axis_53_tvalid,
   input wire m_axis_53_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_53_tdata,

   output wire  m_axis_54_tvalid,
   input wire m_axis_54_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]   m_axis_54_tdata,

   output wire  m_axis_55_tvalid,
   input wire m_axis_55_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_55_tdata,

   output wire  m_axis_56_tvalid,
   input wire m_axis_56_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_56_tdata,

   output wire  m_axis_57_tvalid,
   input wire m_axis_57_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_57_tdata,

   output wire  m_axis_58_tvalid,
   input wire m_axis_58_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_58_tdata,

   output wire  m_axis_59_tvalid,
   input wire m_axis_59_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_59_tdata,

   output wire  m_axis_60_tvalid,
   input wire m_axis_60_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_60_tdata,

   output wire  m_axis_61_tvalid,
   input wire m_axis_61_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_61_tdata,

   output wire  m_axis_62_tvalid,
   input wire m_axis_62_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_62_tdata,

   output wire  m_axis_63_tvalid,
   input wire m_axis_63_tready,
   output wire [(M_AXIS_TDATA_WIDTH_PAD)-1:0]    m_axis_63_tdata




   /*output wire [(C_AXIS_TDATA_WIDTH/8)-1:0]  m_axis_tstrb,
   output wire [(C_AXIS_TDATA_WIDTH/8)-1:0]  m_axis_tkeep,
   output wire [C_NUM_MI_SLOTS-1:0]          m_axis_tlast,
   output wire [(C_AXIS_TID_WIDTH)-1:0]      m_axis_tid,
   output wire [(C_AXIS_TDEST_WIDTH)-1:0]    m_axis_tdest,
   output wire [(C_AXIS_TUSER_WIDTH)-1:0]    m_axis_tuser */

   );

////////////////////////////////////////////////////////////////////////////////
// Functions
////////////////////////////////////////////////////////////////////////////////
`include "axis_infrastructure_v1_1_0.vh"

////////////////////////////////////////////////////////////////////////////////
// Local parameters
////////////////////////////////////////////////////////////////////////////////
//localparam P_TPAYLOAD_WIDTH = f_payload_width(C_AXIS_TDATA_WIDTH, C_AXIS_TID_WIDTH, 
//                                              C_AXIS_TDEST_WIDTH, C_AXIS_TUSER_WIDTH, 
//                                              C_AXIS_SIGNAL_SET);

localparam P_TPAYLOAD_WIDTH = C_AXIS_TDATA_WIDTH;

////////////////////////////////////////////////////////////////////////////////
// Wires/Reg declarations
////////////////////////////////////////////////////////////////////////////////
wire [P_TPAYLOAD_WIDTH-1:0]                   s_axis_tpayload;
wire                                          s_axis_tvalid_i;
wire [C_AXIS_TDATA_WIDTH-1:0]                 m_axis_tdata;
////////////////////////////////////////////////////////////////////////////////
// BEGIN RTL
////////////////////////////////////////////////////////////////////////////////
axis_infrastructure_v1_1_0_util_axis2vector #(
  .C_TDATA_WIDTH    ( C_AXIS_TDATA_WIDTH ) ,
  .C_TID_WIDTH      ( C_AXIS_TID_WIDTH   ) ,
  .C_TDEST_WIDTH    ( C_AXIS_TDEST_WIDTH ) ,
  .C_TUSER_WIDTH    ( C_AXIS_TUSER_WIDTH ) ,
  .C_TPAYLOAD_WIDTH ( P_TPAYLOAD_WIDTH   ) ,
  .C_SIGNAL_SET     ( C_AXIS_SIGNAL_SET  ) 
)
util_axis2vector_0 (
  .TDATA    ( s_axis_tdata[C_AXIS_TDATA_WIDTH - 1:0]) ,
 // .TSTRB    ( s_axis_tstrb    ) ,
//  .TKEEP    ( s_axis_tkeep    ) ,
//  .TLAST    ( s_axis_tlast    ) ,
//  .TID      ( s_axis_tid      ) ,
//  .TDEST    ( s_axis_tdest    ) ,
//  .TUSER    ( s_axis_tuser    ) ,
  .TPAYLOAD ( s_axis_tpayload )
);
//reg aclken = 1'b1;
reg  [C_NUM_MI_SLOTS-1:0] m_ready_d= {C_NUM_MI_SLOTS{1'b0}};

always @(posedge aclk) begin
  if (!aresetn) begin
    m_ready_d <= {C_NUM_MI_SLOTS{1'b0}};
  end else if (aclken) begin
    if (s_axis_tready) begin
      m_ready_d <= {C_NUM_MI_SLOTS{1'b0}};
    end else begin
      m_ready_d <= m_ready_d | ({m_axis_63_tvalid,m_axis_62_tvalid,m_axis_61_tvalid,m_axis_60_tvalid,m_axis_59_tvalid,m_axis_58_tvalid,m_axis_57_tvalid,m_axis_56_tvalid,m_axis_55_tvalid,m_axis_54_tvalid,m_axis_53_tvalid,m_axis_52_tvalid,m_axis_51_tvalid,m_axis_50_tvalid,m_axis_49_tvalid,m_axis_48_tvalid,m_axis_47_tvalid,m_axis_46_tvalid,m_axis_45_tvalid,m_axis_44_tvalid,m_axis_43_tvalid,m_axis_42_tvalid,m_axis_41_tvalid,m_axis_40_tvalid,m_axis_39_tvalid,m_axis_38_tvalid,m_axis_37_tvalid,m_axis_36_tvalid,m_axis_35_tvalid,m_axis_34_tvalid,m_axis_33_tvalid,m_axis_32_tvalid,m_axis_31_tvalid,m_axis_30_tvalid,m_axis_29_tvalid,m_axis_28_tvalid,m_axis_27_tvalid,m_axis_26_tvalid,m_axis_25_tvalid,m_axis_24_tvalid,m_axis_23_tvalid,m_axis_22_tvalid,m_axis_21_tvalid,m_axis_20_tvalid,m_axis_19_tvalid,m_axis_18_tvalid,m_axis_17_tvalid,m_axis_16_tvalid,m_axis_15_tvalid, m_axis_14_tvalid, m_axis_13_tvalid, m_axis_12_tvalid, m_axis_11_tvalid, m_axis_10_tvalid, m_axis_09_tvalid, m_axis_08_tvalid, m_axis_07_tvalid, m_axis_06_tvalid, m_axis_05_tvalid, m_axis_04_tvalid, m_axis_03_tvalid, m_axis_02_tvalid, m_axis_01_tvalid, m_axis_00_tvalid} & {m_axis_63_tready,m_axis_62_tready,m_axis_61_tready,m_axis_60_tready,m_axis_59_tready,m_axis_58_tready,m_axis_57_tready,m_axis_56_tready,m_axis_55_tready,m_axis_54_tready,m_axis_53_tready,m_axis_52_tready,m_axis_51_tready,m_axis_50_tready,m_axis_49_tready,m_axis_48_tready,m_axis_47_tready,m_axis_46_tready,m_axis_45_tready,m_axis_44_tready,m_axis_43_tready,m_axis_42_tready,m_axis_41_tready,m_axis_40_tready,m_axis_39_tready,m_axis_38_tready,m_axis_37_tready,m_axis_36_tready,m_axis_35_tready,m_axis_34_tready,m_axis_33_tready,m_axis_32_tready,m_axis_31_tready,m_axis_30_tready,m_axis_29_tready,m_axis_28_tready,m_axis_27_tready,m_axis_26_tready,m_axis_25_tready,m_axis_24_tready,m_axis_23_tready,m_axis_22_tready,m_axis_21_tready,m_axis_20_tready,m_axis_19_tready,m_axis_18_tready,m_axis_17_tready,m_axis_16_tready,m_axis_15_tready, m_axis_14_tready, m_axis_13_tready, m_axis_12_tready, m_axis_11_tready, m_axis_10_tready, m_axis_09_tready, m_axis_08_tready, m_axis_07_tready, m_axis_06_tready, m_axis_05_tready, m_axis_04_tready, m_axis_03_tready, m_axis_02_tready, m_axis_01_tready, m_axis_00_tready});
    end
  end
end


assign s_axis_tready = (C_AXIS_SIGNAL_SET[0] == 1) ? (&(m_ready_d | {m_axis_63_tready,m_axis_62_tready,m_axis_61_tready,m_axis_60_tready,m_axis_59_tready,m_axis_58_tready,m_axis_57_tready,m_axis_56_tready,m_axis_55_tready,m_axis_54_tready,m_axis_53_tready,m_axis_52_tready,m_axis_51_tready,m_axis_50_tready,m_axis_49_tready,m_axis_48_tready,m_axis_47_tready,m_axis_46_tready,m_axis_45_tready,m_axis_44_tready,m_axis_43_tready,m_axis_42_tready,m_axis_41_tready,m_axis_40_tready,m_axis_39_tready,m_axis_38_tready,m_axis_37_tready,m_axis_36_tready,m_axis_35_tready,m_axis_34_tready,m_axis_33_tready,m_axis_32_tready,m_axis_31_tready,m_axis_30_tready,m_axis_29_tready,m_axis_28_tready,m_axis_27_tready,m_axis_26_tready,m_axis_25_tready,m_axis_24_tready,m_axis_23_tready,m_axis_22_tready,m_axis_21_tready,m_axis_20_tready,m_axis_19_tready,m_axis_18_tready,m_axis_17_tready,m_axis_16_tready,m_axis_15_tready, m_axis_14_tready, m_axis_13_tready, m_axis_12_tready, m_axis_11_tready, m_axis_10_tready, m_axis_09_tready, m_axis_08_tready, m_axis_07_tready, m_axis_06_tready, m_axis_05_tready, m_axis_04_tready, m_axis_03_tready, m_axis_02_tready, m_axis_01_tready, m_axis_00_tready}) & aresetn) : 1'b1;
assign s_axis_tvalid_i = (C_AXIS_SIGNAL_SET[0] == 1) ? (s_axis_tvalid & aresetn) : s_axis_tvalid;

//assign m_axis_tvalid = {C_NUM_MI_SLOTS{s_axis_tvalid_i}} & ~m_ready_d ;
assign {m_axis_63_tvalid,m_axis_62_tvalid,m_axis_61_tvalid,m_axis_60_tvalid,m_axis_59_tvalid,m_axis_58_tvalid,m_axis_57_tvalid,m_axis_56_tvalid,m_axis_55_tvalid,m_axis_54_tvalid,m_axis_53_tvalid,m_axis_52_tvalid,m_axis_51_tvalid,m_axis_50_tvalid,m_axis_49_tvalid,m_axis_48_tvalid,m_axis_47_tvalid,m_axis_46_tvalid,m_axis_45_tvalid,m_axis_44_tvalid,m_axis_43_tvalid,m_axis_42_tvalid,m_axis_41_tvalid,m_axis_40_tvalid,m_axis_39_tvalid,m_axis_38_tvalid,m_axis_37_tvalid,m_axis_36_tvalid,m_axis_35_tvalid,m_axis_34_tvalid,m_axis_33_tvalid,m_axis_32_tvalid,m_axis_31_tvalid,m_axis_30_tvalid,m_axis_29_tvalid,m_axis_28_tvalid,m_axis_27_tvalid,m_axis_26_tvalid,m_axis_25_tvalid,m_axis_24_tvalid,m_axis_23_tvalid,m_axis_22_tvalid,m_axis_21_tvalid,m_axis_20_tvalid,m_axis_19_tvalid,m_axis_18_tvalid,m_axis_17_tvalid,m_axis_16_tvalid,m_axis_15_tvalid, m_axis_14_tvalid, m_axis_13_tvalid, m_axis_12_tvalid, m_axis_11_tvalid, m_axis_10_tvalid, m_axis_09_tvalid, m_axis_08_tvalid, m_axis_07_tvalid, m_axis_06_tvalid, m_axis_05_tvalid, m_axis_04_tvalid, m_axis_03_tvalid, m_axis_02_tvalid, m_axis_01_tvalid, m_axis_00_tvalid} = {C_NUM_MI_SLOTS{s_axis_tvalid_i}} & ~m_ready_d ;



genvar mi;
generate
  for (mi = 0; mi < C_NUM_MI_SLOTS; mi = mi + 1) begin : MI_SLOT
    axis_infrastructure_v1_1_0_util_vector2axis #(
      .C_TDATA_WIDTH    ( C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) ,
      .C_TID_WIDTH      ( C_AXIS_TID_WIDTH   ) ,
      .C_TDEST_WIDTH    ( C_AXIS_TDEST_WIDTH ) ,
      .C_TUSER_WIDTH    ( C_AXIS_TUSER_WIDTH ) ,
      .C_TPAYLOAD_WIDTH ( P_TPAYLOAD_WIDTH  /  C_NUM_MI_SLOTS) ,
      .C_SIGNAL_SET     ( C_AXIS_SIGNAL_SET  ) 
    )
    util_vector2axis (
      .TPAYLOAD ( s_axis_tpayload[mi*(C_AXIS_TDATA_WIDTH/C_NUM_MI_SLOTS)+: (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)]),                                               
      .TDATA    ( m_axis_tdata[mi*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) +: (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)] ) 
      //.TSTRB    ( m_axis_tstrb[mi*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)/8+:C_AXIS_TDATA_WIDTH/8] ) ,
      //.TKEEP    ( m_axis_tkeep[mi*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)/8+:C_AXIS_TDATA_WIDTH/8] ) ,
      //.TLAST    ( m_axis_tlast[mi+:1]                                         ) ,
      //.TID      ( m_axis_tid  [mi*C_AXIS_TID_WIDTH+:C_AXIS_TID_WIDTH]         ) ,
      //.TDEST    ( m_axis_tdest[mi*C_AXIS_TDEST_WIDTH+:C_AXIS_TDEST_WIDTH]     ) ,
      //.TUSER    ( m_axis_tuser[mi*C_AXIS_TUSER_WIDTH+:C_AXIS_TUSER_WIDTH]     ) 
    );
  end
endgenerate



assign m_axis_00_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[1*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
//assign m_axis_00_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};



if (C_NUM_MI_SLOTS > 1) begin
   assign m_axis_01_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[2*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 1 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
   //assign m_axis_01_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 2) begin
   assign m_axis_02_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[3*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 2 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
   //assign m_axis_02_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 3) begin
   assign m_axis_03_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[4*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 3 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
   //assign m_axis_03_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 4) begin
   assign m_axis_04_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[5*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 4 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
   //assign m_axis_04_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 5) begin
   assign m_axis_05_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[6*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 5 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
  // assign m_axis_05_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 6) begin
   assign m_axis_06_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[7*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 6 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
 //  assign m_axis_06_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 7) begin
   assign m_axis_07_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[8*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 7 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
  // assign m_axis_07_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 8) begin
   assign m_axis_08_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[9*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 8 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
  // assign m_axis_08_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 9) begin
   assign m_axis_09_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[10*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 9 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
  // assign m_axis_09_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end

if (C_NUM_MI_SLOTS > 10) begin
   assign m_axis_10_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[11*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 10 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
  // assign m_axis_10_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 11) begin
   assign m_axis_11_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[12*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 11 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
 //  assign m_axis_11_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 12) begin
   assign m_axis_12_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[13*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 12 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
 //  assign m_axis_12_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 13) begin
   assign m_axis_13_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[14*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 13 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
 //  assign m_axis_13_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 14) begin
   assign m_axis_14_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[15*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 14 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
 //  assign m_axis_14_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 15) begin
   assign m_axis_15_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[16*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 15 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
 //  assign m_axis_15_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 16) begin
   assign m_axis_16_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[17*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 16 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
  // assign m_axis_16_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 17) begin
   assign m_axis_17_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[18*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 17 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
 //  assign m_axis_17_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 18) begin
   assign m_axis_18_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[19*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 18 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
 //  assign m_axis_18_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 19) begin
   assign m_axis_19_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[20*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 19 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
 //  assign m_axis_19_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 20) begin
   assign m_axis_20_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[21*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 20 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
 //  assign m_axis_20_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 21) begin
   assign m_axis_21_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[22*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 21 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
 //  assign m_axis_21_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 22) begin
   assign m_axis_22_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[23*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 22 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
 //  assign m_axis_22_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 23) begin
   assign m_axis_23_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[24*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 23 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
 //  assign m_axis_23_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 24) begin
   assign m_axis_24_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[25*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 24 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
 //  assign m_axis_24_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 25) begin
   assign m_axis_25_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[26*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 25 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
//   assign m_axis_25_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 26) begin
   assign m_axis_26_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[27*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 26 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
 //  assign m_axis_26_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 27) begin
   assign m_axis_27_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[28*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 27 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
 //  assign m_axis_27_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 28) begin
   assign m_axis_28_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[29*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 28 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
 //  assign m_axis_28_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 29) begin
   assign m_axis_29_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[30*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 29 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
 //  assign m_axis_29_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 30) begin
   assign m_axis_30_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[31*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 30 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
 //  assign m_axis_30_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 31) begin
   assign m_axis_31_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[32*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 31 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
 //  assign m_axis_31_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 32) begin
   assign m_axis_32_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[33*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 32 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
  // assign m_axis_32_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 33) begin
   assign m_axis_33_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[34*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 33 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
  // assign m_axis_33_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 34) begin
   assign m_axis_34_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[35*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 34 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
  // assign m_axis_34_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 35) begin
   assign m_axis_35_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[36*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 35 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
 //  assign m_axis_35_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 36) begin
   assign m_axis_36_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[37*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 36 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
 //  assign m_axis_36_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 37) begin
   assign m_axis_37_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[38*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 37 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
  // assign m_axis_37_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 38) begin
   assign m_axis_38_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[39*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 38 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
  // assign m_axis_38_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 39) begin
   assign m_axis_39_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[40*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 39 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
  // assign m_axis_39_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 40) begin
   assign m_axis_40_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[41*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 40 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
 //  assign m_axis_40_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 41) begin
   assign m_axis_41_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[42*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 41 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
 //  assign m_axis_41_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 42) begin
   assign m_axis_42_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[43*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 42 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
//   assign m_axis_42_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 43) begin
   assign m_axis_43_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[44*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 43 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
 //  assign m_axis_43_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 44) begin
   assign m_axis_44_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[45*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 44 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
 //  assign m_axis_44_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 45) begin
   assign m_axis_45_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[46*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 45 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
  // assign m_axis_45_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 46) begin
   assign m_axis_46_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[47*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 46 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
 //  assign m_axis_46_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 47) begin
   assign m_axis_47_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[48*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 47 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
   //assign m_axis_47_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 48) begin
   assign m_axis_48_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[49*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 48 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
  /// assign m_axis_48_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 49) begin
   assign m_axis_49_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[50*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 49 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
   //assign m_axis_49_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 50) begin
   assign m_axis_50_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[51*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 50 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
  // assign m_axis_50_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 51) begin
   assign m_axis_51_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[52*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 51 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
   //assign m_axis_51_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 52) begin
   assign m_axis_52_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[53*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 52 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
   //assign m_axis_52_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 53) begin
   assign m_axis_53_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[54*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 53 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
   //assign m_axis_53_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 54) begin
   assign m_axis_54_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[55*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 54 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
  // assign m_axis_54_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 55) begin
   assign m_axis_55_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[56*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 55 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
   //assign m_axis_55_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 56) begin
   assign m_axis_56_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[57*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 56 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
   //assign m_axis_56_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 57) begin
   assign m_axis_57_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[58*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 57 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
   //assign m_axis_57_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 58) begin
   assign m_axis_58_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[59*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 58 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
  // assign m_axis_58_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 59) begin
   assign m_axis_59_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[60*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 59 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
  // assign m_axis_59_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 60) begin
   assign m_axis_60_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[61*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 60 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
  // assign m_axis_60_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 61) begin
   assign m_axis_61_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[62*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 61 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
  // assign m_axis_61_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 62) begin
   assign m_axis_62_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[63*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 62 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
 //  assign m_axis_62_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


if (C_NUM_MI_SLOTS > 63) begin
   assign m_axis_63_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[64*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 63 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
  // assign m_axis_63_tdata[M_AXIS_TDATA_WIDTH_PAD - 1 : (C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS)] =  {(M_AXIS_TDATA_WIDTH_PAD - (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)){1'b0}};
end


/*

if (C_NUM_MI_SLOTS > 1) begin
   assign m_axis_01_tdata[(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 0 ] = m_axis_tdata[2*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 1 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_01_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 2) begin
   assign m_axis_02_tdata = m_axis_tdata[3*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 2 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_02_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 3) begin
   assign m_axis_03_tdata = m_axis_tdata[4*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 3 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_03_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 4) begin
   assign m_axis_04_tdata = m_axis_tdata[5*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 4 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_04_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 5) begin
   assign m_axis_05_tdata = m_axis_tdata[6*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 5 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_05_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 6) begin
   assign m_axis_06_tdata = m_axis_tdata[7*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 6 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_06_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 7) begin
   assign m_axis_07_tdata = m_axis_tdata[8*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 7 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_07_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 8) begin
   assign m_axis_08_tdata = m_axis_tdata[9*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 8 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_08_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 9) begin
   assign m_axis_09_tdata = m_axis_tdata[10*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 9 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_09_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 10) begin
   assign m_axis_10_tdata = m_axis_tdata[11*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 10 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_10_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 11) begin
   assign m_axis_11_tdata = m_axis_tdata[12*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 11 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_11_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 12) begin
   assign m_axis_12_tdata = m_axis_tdata[13*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 12 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_12_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 13) begin
   assign m_axis_13_tdata = m_axis_tdata[14*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 13 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_13_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 14) begin
   assign m_axis_14_tdata = m_axis_tdata[15*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 14 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_14_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 15) begin
   assign m_axis_15_tdata = m_axis_tdata[16*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 15 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_15_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 16) begin
   assign m_axis_16_tdata = m_axis_tdata[17*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 16 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_16_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 17) begin
   assign m_axis_17_tdata = m_axis_tdata[18*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 17 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_17_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 18) begin
   assign m_axis_18_tdata = m_axis_tdata[19*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 18 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_18_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 19) begin
   assign m_axis_19_tdata = m_axis_tdata[20*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 19 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_19_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 20) begin
   assign m_axis_20_tdata = m_axis_tdata[21*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 20 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_20_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 21) begin
   assign m_axis_21_tdata = m_axis_tdata[22*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 21 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_21_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 22) begin
   assign m_axis_22_tdata = m_axis_tdata[23*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 22 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_22_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 23) begin
   assign m_axis_23_tdata = m_axis_tdata[24*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 23 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_23_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 24) begin
   assign m_axis_24_tdata = m_axis_tdata[25*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 24 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_24_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 25) begin
   assign m_axis_25_tdata = m_axis_tdata[26*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 25 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_25_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 26) begin
   assign m_axis_26_tdata = m_axis_tdata[27*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 26 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_26_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 27) begin
   assign m_axis_27_tdata = m_axis_tdata[28*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 27 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_27_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 28) begin
   assign m_axis_28_tdata = m_axis_tdata[29*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 28 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_28_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 29) begin
   assign m_axis_29_tdata = m_axis_tdata[30*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 29 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_29_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 30) begin
   assign m_axis_30_tdata = m_axis_tdata[31*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 30 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_30_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 31) begin
   assign m_axis_31_tdata = m_axis_tdata[32*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 31 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_31_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 32) begin
   assign m_axis_32_tdata = m_axis_tdata[33*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 32 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_32_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 33) begin
   assign m_axis_33_tdata = m_axis_tdata[34*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 33 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_33_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 34) begin
   assign m_axis_34_tdata = m_axis_tdata[35*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 34 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_34_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 35) begin
   assign m_axis_35_tdata = m_axis_tdata[36*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 35 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_35_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 36) begin
   assign m_axis_36_tdata = m_axis_tdata[37*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 36 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_36_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 37) begin
   assign m_axis_37_tdata = m_axis_tdata[38*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 37 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_37_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 38) begin
   assign m_axis_38_tdata = m_axis_tdata[39*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 38 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_38_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 39) begin
   assign m_axis_39_tdata = m_axis_tdata[40*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 39 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_39_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 40) begin
   assign m_axis_40_tdata = m_axis_tdata[41*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 40 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_40_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 41) begin
   assign m_axis_41_tdata = m_axis_tdata[42*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 41 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_41_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 42) begin
   assign m_axis_42_tdata = m_axis_tdata[43*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 42 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_42_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 43) begin
   assign m_axis_43_tdata = m_axis_tdata[44*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 43 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_43_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 44) begin
   assign m_axis_44_tdata = m_axis_tdata[45*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 44 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_44_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 45) begin
   assign m_axis_45_tdata = m_axis_tdata[46*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 45 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_45_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 46) begin
   assign m_axis_46_tdata = m_axis_tdata[47*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 46 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_46_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 47) begin
   assign m_axis_47_tdata = m_axis_tdata[48*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 47 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_47_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 48) begin
   assign m_axis_48_tdata = m_axis_tdata[49*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 48 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_48_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 49) begin
   assign m_axis_49_tdata = m_axis_tdata[50*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 49 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_49_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 50) begin
   assign m_axis_50_tdata = m_axis_tdata[51*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 50 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_50_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 51) begin
   assign m_axis_51_tdata = m_axis_tdata[52*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 51 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_51_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 52) begin
   assign m_axis_52_tdata = m_axis_tdata[53*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 52 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_52_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 53) begin
   assign m_axis_53_tdata = m_axis_tdata[54*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 53 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_53_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 54) begin
   assign m_axis_54_tdata = m_axis_tdata[55*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 54 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_54_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 55) begin
   assign m_axis_55_tdata = m_axis_tdata[56*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 55 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_55_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 56) begin
   assign m_axis_56_tdata = m_axis_tdata[57*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 56 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_56_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 57) begin
   assign m_axis_57_tdata = m_axis_tdata[58*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 57 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_57_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 58) begin
   assign m_axis_58_tdata = m_axis_tdata[59*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 58 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_58_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 59) begin
   assign m_axis_59_tdata = m_axis_tdata[60*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 59 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_59_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 60) begin
   assign m_axis_60_tdata = m_axis_tdata[61*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 60 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_60_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 61) begin
   assign m_axis_61_tdata = m_axis_tdata[62*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 61 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_61_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 62) begin
   assign m_axis_62_tdata = m_axis_tdata[63*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 62 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_62_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

if (C_NUM_MI_SLOTS > 63) begin
   assign m_axis_63_tdata = m_axis_tdata[64*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 63 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)];
end else begin
   assign m_axis_63_tdata = {(C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS){1'b0}};
end

*/
/*
assign m_axis_01_tdata = m_axis_tdata[2*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 1 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)]
assign m_axis_02_tdata = m_axis_tdata[3*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 2 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)]
assign m_axis_03_tdata = m_axis_tdata[4*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 3 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)]

assign m_axis_04_tdata = m_axis_tdata[5*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 4 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)]
assign m_axis_05_tdata = m_axis_tdata[6*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 5 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)]
assign m_axis_06_tdata = m_axis_tdata[7*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 6 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)]
assign m_axis_07_tdata = m_axis_tdata[8*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 7 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)]
assign m_axis_08_tdata = m_axis_tdata[9*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 8 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)]
assign m_axis_09_tdata = m_axis_tdata[10*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 9 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)]
assign m_axis_10_tdata = m_axis_tdata[11*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 10 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)]
assign m_axis_11_tdata = m_axis_tdata[12*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 11 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)]
assign m_axis_12_tdata = m_axis_tdata[13*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 12 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)]
assign m_axis_13_tdata = m_axis_tdata[14*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 13 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)]
assign m_axis_14_tdata = m_axis_tdata[15*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 14 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)]
assign m_axis_15_tdata = m_axis_tdata[16*(C_AXIS_TDATA_WIDTH / C_NUM_MI_SLOTS) - 1 : 15 * (C_AXIS_TDATA_WIDTH/ C_NUM_MI_SLOTS)]
*/
endmodule // axis_broadcaster_top

`default_nettype wire


