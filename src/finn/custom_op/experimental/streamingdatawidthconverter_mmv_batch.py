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

import os
import numpy as np
import math
import warnings
from finn.custom_op.fpgadataflow.streamingdatawidthconverter_batch import StreamingDataWidthConverter_Batch
from finn.core.datatype import DataType
from onnx import TensorProto, helper
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy
from finn.util.basic import roundup_to_integer_multiple
# does not do anything at the ONNX node-by-node level, and input-output
# tensor shapes are the same. performs data width conversion at the rtlsim level


class StreamingDataWidthConverter_MMV_Batch(StreamingDataWidthConverter_Batch):
    """Class that extends StreamingDataWidthConverter_Batch with MMV support."""

    def __init__(self, onnx_node):
        super().__init__(onnx_node)

    def get_nodeattr_types(self):
        my_attrs = {
            "MMV" : ("i", False, 1),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def code_generation_ipi(self):
        impl_style = self.get_nodeattr("impl_style")
        mmv_value = self.get_nodeattr("MMV")
        # for MMV=1 fall back to original implementation
        if mmv_value == 1:
            return super().code_generation_ipi()
        #force HLS implementation when MMV>1
        if self.get_nodeattr("MMV") > 1:
            self.set_nodeattr("impl_style", "hls")
            impl_style = "hls"

        cmd = []
        node_name = self.onnx_node.name
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
            "-vlnv xilinx.com:interface:axis_rtl:1.0 /%s/%s"
            % (node_name,  din_name)
        )
        for m in range(mmv_value): 
            cmd.append(
                "create_bd_cell -type ip -vlnv %s /%s/%s_%d"
                % (self.get_nodeattr("ip_vlnv"), node_name, node_name, m)
            )               
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s_%d/%s]"
                % (node_name, rst_name, node_name, node_name, m, rst_name)
            )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s_%d/%s]"
                % (node_name, clk_name, node_name, node_name, m, clk_name)
            )
        # instantiate splitter inputs
        cmd.append("create_bd_cell -type ip -vlnv user.org:user:axis_split_core:1.0 %s/axis_splitter_input" % (node_name))
        cmd.append("set_property -dict [list CONFIG.S_AXIS_TDATA_WIDTH_PAD {%d} CONFIG.C_AXIS_TDATA_WIDTH {%d} CONFIG.M_AXIS_TDATA_WIDTH_PAD {%d} CONFIG.C_NUM_MI_SLOTS {%d}] [get_bd_cells %s/axis_splitter_input]" % (roundup_to_integer_multiple(self.get_instream_width() * mmv_value, 8), self.get_instream_width() * mmv_value, self.get_instream_width(), mmv_value, node_name))

        # instantiate combiner
        cmd.append("create_bd_cell -type ip -vlnv user.org:user:axis_combiner_v1_1_19_top:1.0 %s/axis_combiner_output" % (node_name))
        cmd.append("set_property -dict [list CONFIG.C_AXIS_TDATA_WIDTH {%d} CONFIG.C_AXIS_SIGNAL_SET {0x00000003} CONFIG.C_NUM_SI_SLOTS {%d}] [get_bd_cells %s/axis_combiner_output]" % (self.get_outstream_width(), mmv_value, node_name)) 

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
        # connect combiner to output of block
        cmd.append(
            "connect_bd_intf_net [get_bd_intf_pins %s/axis_combiner_output/m_axis] "
            "[get_bd_intf_pins %s/%s]"
            %(node_name, node_name, dout_name)
        )
        # connect input of block to splitter
        cmd.append(
            "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
            "[get_bd_intf_pins %s/axis_splitter_input/s_axis]"
            % (node_name, din_name, node_name)
        )
        for m in range(mmv_value):
            cmd.append(
                "connect_bd_intf_net [get_bd_intf_pins %s/%s_%d/%s] "
                "[get_bd_intf_pins %s/axis_splitter_input/m_axis_%02d]" % (node_name, node_name, m, din_name, node_name, m)
            ) 
            cmd.append(
                "connect_bd_intf_net [get_bd_intf_pins %s/%s_%d/%s] "
                "[get_bd_intf_pins %s/axis_combiner_output/s_axis_%02d]" % (node_name, node_name, m, dout_name, node_name, m)
            ) 
        return cmd  

    def lut_estimation(self):
        """Calculates resource estimations for LUTs"""
        mmv_value = self.get_nodeattr("MMV")
        return mmv_value*super().lut_estimation()

