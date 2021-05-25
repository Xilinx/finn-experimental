# Copyright (c) 2020, Xilinx
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from math import ceil, log2
import textwrap
import os
import warnings
import numpy as np

from onnx import TensorProto, helper
from finn.core.datatype import DataType
from finn.custom_op.fpgadataflow.thresholding_batch import Thresholding_Batch
from finn.util.basic import (
    interleave_matrix_outer_dim_from_partitions,
    roundup_to_integer_multiple,
)
from finn.util.data_packing import (
    npy_to_rtlsim_input,
    numpy_to_hls_code,
    rtlsim_output_to_npy,
    pack_innermost_dim_as_hex_string,
)
from . import templates

# ONNX i/o tensor shape assumptions for Thresholding:
# input 0 is the input tensor, shape (..., NumChannels)
# input 1 is the threshold tensor, shape (NumChannels, n_thres)
# output 0 is the output tensor, shape (..., NumChannels) - same as input
# the ... here can be any shape (representing groups of vectors)


class Thresholding_MMV_Batch(Thresholding_Batch):
    """Class that corresponds to finn-hls Thresholding_Batch function."""

    def __init__(self, onnx_node):
        super().__init__(onnx_node)
        self.decoupled_wrapper = templates.decoupled_wrapper

    def get_nodeattr_types(self):
        my_attrs = {
            "MMV" : ("i", False, 1),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def bram_estimation(self):
        mmv = self.get_nodeattr("MMV")
        return mmv * super().bram_estimation()

    def lut_estimation(self):
        mmv = self.get_nodeattr("MMV")
        return mmv * super().lut_estimation()

    def code_generation_ipi(self):
        cmd = []
        # add streamer if needed

        mmv_value = self.get_nodeattr("MMV")
        if mmv_value > 1:
           self.set_nodeattr("mem_mode", "const")
           #assert mem_mode == "decoupled"
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "decoupled":
            node_name = self.onnx_node.name
            runtime_writable = self.get_nodeattr("runtime_writeable_weights") == 1
            # create a hierarchy for this layer, with the same port names
            clk_name = self.get_verilog_top_module_intf_names()["clk"][0]
            rst_name = self.get_verilog_top_module_intf_names()["rst"][0]
            dout_name = self.get_verilog_top_module_intf_names()["m_axis"][0]
            din_name = self.get_verilog_top_module_intf_names()["s_axis"][0]
            cmd.append("create_bd_cell -type hier %s" % node_name)
            cmd.append("create_bd_pin -dir I -type clk /%s/%s" % (node_name, clk_name))
            cmd.append("create_bd_pin -dir I -type rst /%s/%s" % (node_name, rst_name))
            cmd.append(
                "create_bd_intf_pin -mode Master "
                "-vlnv xilinx.com:interface:axis_rtl:1.0 /%s/%s"
                % (node_name, dout_name)
            )
            cmd.append(
                "create_bd_intf_pin -mode Slave "
                "-vlnv xilinx.com:interface:axis_rtl:1.0 /%s/%s" % (node_name, din_name)
            )

            # instantiate a streamer and connect it to the HLS IP
            strm_vlnv = "xilinx.com:user:memstream:1.0"
            strm_inst = node_name + "_wstrm"
            cmd.append(
                "create_bd_cell -type ip -vlnv %s /%s/%s"
                % (strm_vlnv, node_name, strm_inst)
            )
            cmd.append(
                "set_property -dict [list "
                "CONFIG.NSTREAMS {1} "
                "CONFIG.MEM_DEPTH {%d} "
                "CONFIG.MEM_WIDTH {%d} "
                "CONFIG.MEM_INIT {%s} "
                "CONFIG.RAM_STYLE {%s} "
                "CONFIG.STRM0_DEPTH {%d} "
                "CONFIG.STRM0_WIDTH {%d} "
                "CONFIG.STRM0_OFFSET {0} "
                "] [get_bd_cells /%s/%s]"
                % (
                    self.calc_tmem(),
                    self.get_weightstream_width_padded(),
                    self.get_nodeattr("code_gen_dir_ipgen") + "/",
                    self.get_nodeattr("ram_style"),
                    self.calc_tmem(),
                    self.get_weightstream_width_padded(),
                    node_name,
                    strm_inst,
                )
            )
            if mmv_value == 1:
               # instantiate the hls ip
               cmd.append(
                 "create_bd_cell -type ip -vlnv %s /%s/%s"
                 % (self.get_nodeattr("ip_vlnv"), node_name, node_name)
               )
               cmd.append(
                  "connect_bd_intf_net [get_bd_intf_pins %s/%s/m_axis_0] "
                  "[get_bd_intf_pins %s/%s/weights_V_V]"
                  % (node_name, strm_inst, node_name, node_name)
               )
               cmd.append(
                  "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/%s]"
                  % (node_name, rst_name, node_name, node_name, rst_name)
               )
               cmd.append(
                  "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/%s]"
                  % (node_name, clk_name, node_name, node_name, clk_name)
               )
               cmd.append(
                  "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
                  "[get_bd_intf_pins %s/%s/%s]"
                  % (node_name, din_name, node_name, node_name, din_name)
               )
               cmd.append(
                 "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
                 "[get_bd_intf_pins %s/%s/%s]"
                 % (node_name, dout_name, node_name, node_name, dout_name)
               )
      
            else :
               # instantiate HLS IPs
               for i in range(mmv_value):
                  cmd.append(
                    "create_bd_cell -type ip -vlnv %s /%s/%s_%d"
                    % (self.get_nodeattr("ip_vlnv"), node_name, node_name, i)
                  )
                  cmd.append(
                    "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s_%d/%s]"
                    % (node_name, rst_name, node_name, node_name, i, rst_name)
                  )
                  cmd.append(
                    "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s_%d/%s]"
                    % (node_name, clk_name, node_name, node_name, i, clk_name)
                  )
               # instantiate splitter inputs
               cmd.append("create_bd_cell -type ip -vlnv user.org:user:axis_split_core:1.0 %s/axis_splitter_input" % (node_name))
               cmd.append("set_property -dict [list CONFIG.S_AXIS_TDATA_WIDTH_PAD {%d} CONFIG.C_AXIS_TDATA_WIDTH {%d} CONFIG.M_AXIS_TDATA_WIDTH_PAD {%d} CONFIG.C_NUM_MI_SLOTS {%d}] [get_bd_cells %s/axis_splitter_input]" % (roundup_to_integer_multiple(self.get_instream_width() * mmv_value, 8), self.get_instream_width() * mmv_value, self.get_instream_width(), mmv_value, node_name))
               # instantiate broadcaster weights
               cmd.append("create_bd_cell -type ip -vlnv user.org:user:extend_broadcaster2:1.0 %s/axis_broadcaster_weight" % (node_name))
               cmd.append("set_property -dict [list CONFIG.C_AXIS_TDATA_WIDTH {%d} CONFIG.C_NUM_MI_SLOTS {%s}] [get_bd_cells %s/axis_broadcaster_weight]" % (self.get_weightstream_width_padded(), mmv_value, node_name))

               # instantiate combiner
               cmd.append("create_bd_cell -type ip -vlnv user.org:user:axis_combiner_v1_1_19_top:1.0 %s/axis_combiner_output" % (node_name))
               cmd.append("set_property -dict [list CONFIG.C_AXIS_TDATA_WIDTH {%d} CONFIG.C_AXIS_SIGNAL_SET {0x00000003} CONFIG.C_NUM_SI_SLOTS {%d}] [get_bd_cells %s/axis_combiner_output]" % (self.get_outstream_width(), mmv_value, node_name)) 

               # connect input of block to splitter
               cmd.append(
                  "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
                  "[get_bd_intf_pins %s/axis_splitter_input/s_axis]"
                  % (node_name, din_name, node_name)
               )
               # connect output of streamer to splitter
               cmd.append(
                  "connect_bd_intf_net [get_bd_intf_pins %s/%s/m_axis_0] "
                  "[get_bd_intf_pins %s/axis_broadcaster_weight/s_axis]"
                  % (node_name, strm_inst, node_name)
               )


               # connect splitter to PEs
               for i in range(mmv_value):
                  cmd.append(
                     "connect_bd_intf_net [get_bd_intf_pins %s/axis_splitter_input/m_axis_%02d] "
                     "[get_bd_intf_pins %s/%s_%d/%s]"
                     % (node_name, i, node_name, node_name, i, din_name)
                  )
               # connect broadcaster to PEs
               for i in range(mmv_value):
                  cmd.append(
                     "connect_bd_intf_net [get_bd_intf_pins %s/axis_broadcaster_weight/m_axis_%02d] "
                     "[get_bd_intf_pins %s/%s_%d/weights_V_V]"
                     % (node_name, i, node_name, node_name, i)
                  )
               # connect PEs to combiner
               for i in range(mmv_value):
                  cmd.append("connect_bd_intf_net [get_bd_intf_pins %s/%s_%d/out_V_V] [get_bd_intf_pins %s/axis_combiner_output/s_axis_%02d]" % (node_name, node_name, i, node_name, i))

               # connect combiner to output of block
               cmd.append(
                   "connect_bd_intf_net [get_bd_intf_pins %s/axis_combiner_output/m_axis] "
                   "[get_bd_intf_pins %s/%s]"
                   %(node_name, node_name, dout_name)
               )
               # connect clk and resets
               cmd.append(
                  "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/axis_combiner_output/aresetn]"
                  % (node_name, rst_name, node_name)
               )
               cmd.append(
                 "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/axis_combiner_output/aclk]"
                  % (node_name, clk_name, node_name)
               ) 

               # connect clk and reset - weight broadcaster
               cmd.append(
                 "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/axis_broadcaster_weight/aresetn]"
                  % (node_name, rst_name, node_name)
               )
               cmd.append(
                 "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/axis_broadcaster_weight/aclk]"
                 % (node_name, clk_name, node_name)
               ) 
               cmd.append(
                  "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/axis_splitter_input/aresetn]"
                  % (node_name, rst_name, node_name)
               )
               cmd.append(
                  "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/axis_splitter_input/aclk]"
                   % (node_name, clk_name, node_name)
               )
            cmd.append(
                 "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/aresetn]"
                 % (node_name, rst_name, node_name, strm_inst)
            )
            cmd.append(
               "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/aclk]"
               % (node_name, clk_name, node_name, strm_inst)
            )

            if runtime_writable:
                # expose axi lite interface for writeable weights
                axilite_name = self.get_verilog_top_module_intf_names()["axilite"][0]
                cmd.append(
                    "create_bd_intf_pin -mode Slave "
                    "-vlnv xilinx.com:interface:aximm_rtl:1.0 /%s/%s"
                    % (node_name, axilite_name)
                )
                cmd.append(
                    "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
                    "[get_bd_intf_pins %s/%s/%s]"
                    % (node_name, axilite_name, node_name, strm_inst, axilite_name)
                )
                # TODO calculate and pass in segment size here
                cmd.append("assign_bd_address")
            cmd.append("save_bd_design")
        elif mem_mode == "const":
            if mmv_value > 1 :

               # create a hierarchy for this layer, with the same port names
               clk_name = self.get_verilog_top_module_intf_names()["clk"][0]
               rst_name = self.get_verilog_top_module_intf_names()["rst"][0]
               dout_name = self.get_verilog_top_module_intf_names()["m_axis"][0]
               din_name = self.get_verilog_top_module_intf_names()["s_axis"][0]
               cmd.append("create_bd_cell -type hier %s" % node_name)
               cmd.append("create_bd_pin -dir I -type clk /%s/%s" % (node_name, clk_name))
               cmd.append("create_bd_pin -dir I -type rst /%s/%s" % (node_name, rst_name))
               cmd.append(
                 "create_bd_intf_pin -mode Master "
                 "-vlnv xilinx.com:interface:axis_rtl:1.0 /%s/%s"
                 % (node_name, dout_name)
               )
               cmd.append(
                "create_bd_intf_pin -mode Slave "
                "-vlnv xilinx.com:interface:axis_rtl:1.0 /%s/%s" % (node_name, din_name)
               )

               # instantiate HLS IPs
               for i in range(mmv_value):
                  cmd.append(
                    "create_bd_cell -type ip -vlnv %s /%s/%s_%d"
                    % (self.get_nodeattr("ip_vlnv"), node_name, node_name, i)
                  )
                  cmd.append(
                    "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s_%d/%s]"
                    % (node_name, rst_name, node_name, node_name, i, rst_name)
                  )
                  cmd.append(
                    "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s_%d/%s]"
                    % (node_name, clk_name, node_name, node_name, i, clk_name)
                  )
               # instantiate splitter inputs
               cmd.append("create_bd_cell -type ip -vlnv user.org:user:axis_split_core:1.0 %s/axis_splitter_input" % (node_name))
               cmd.append("set_property -dict [list CONFIG.S_AXIS_TDATA_WIDTH_PAD {%d} CONFIG.C_AXIS_TDATA_WIDTH {%d} CONFIG.M_AXIS_TDATA_WIDTH_PAD {%d} CONFIG.C_NUM_MI_SLOTS {%d}] [get_bd_cells %s/axis_splitter_input]" % (roundup_to_integer_multiple(self.get_instream_width() * mmv_value, 8), self.get_instream_width() * mmv_value, self.get_instream_width(), mmv_value, node_name))

               # instantiate combiner
               cmd.append("create_bd_cell -type ip -vlnv user.org:user:axis_combiner_v1_1_19_top:1.0 %s/axis_combiner_output" % (node_name))
               cmd.append("set_property -dict [list CONFIG.C_AXIS_TDATA_WIDTH {%d} CONFIG.C_AXIS_SIGNAL_SET {0x00000003} CONFIG.C_NUM_SI_SLOTS {%d}] [get_bd_cells %s/axis_combiner_output]" % (self.get_outstream_width(), mmv_value, node_name)) 

               # connect input of block to splitter
               cmd.append(
                  "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
                  "[get_bd_intf_pins %s/axis_splitter_input/s_axis]"
                  % (node_name, din_name, node_name)
               )
                # connect splitter to PEs
               for i in range(mmv_value):
                  cmd.append(
                     "connect_bd_intf_net [get_bd_intf_pins %s/axis_splitter_input/m_axis_%02d] "
                     "[get_bd_intf_pins %s/%s_%d/%s]"
                     % (node_name, i, node_name, node_name, i, din_name)
                  )
               # connect PEs to combiner
               for i in range(mmv_value):
                  cmd.append("connect_bd_intf_net [get_bd_intf_pins %s/%s_%d/out_V_V] [get_bd_intf_pins %s/axis_combiner_output/s_axis_%02d]" % (node_name, node_name, i, node_name, i))

               # connect combiner to output of block
               cmd.append(
                   "connect_bd_intf_net [get_bd_intf_pins %s/axis_combiner_output/m_axis] "
                   "[get_bd_intf_pins %s/%s]"
                   %(node_name, node_name, dout_name)
               )
               # connect clk and resets
               cmd.append(
                  "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/axis_combiner_output/aresetn]"
                  % (node_name, rst_name, node_name)
               )
               cmd.append(
                 "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/axis_combiner_output/aclk]"
                  % (node_name, clk_name, node_name)
               ) 
               cmd.append(
                  "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/axis_splitter_input/aresetn]"
                  % (node_name, rst_name, node_name)
               )
               cmd.append(
                  "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/axis_splitter_input/aclk]"
                   % (node_name, clk_name, node_name)
               )
            else: 
               # base class impl sufficient for const mode
               return super().code_generation_ipi()
        else:
            raise Exception("Unrecognized mem_mode for Thresholding_Batch")
        return cmd

