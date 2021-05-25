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

import math
import numpy as np

from finn.core.datatype import DataType
from finn.custom_op.fpgadataflow.convolutioninputgenerator import ConvolutionInputGenerator
from finn.custom_op.general.im2col import compute_conv_output_dim
from onnx import TensorProto, helper
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy
from finn.util.basic import (
    roundup_to_integer_multiple,
)
# ONNX i/o tensor shape assumptions for ConvolutionInputGenerator:
# input 0 is the input tensor, shape NHWC = (1, IFMDim, IFMDim, IFMChannels)
# output 0 is the output tensor, shape NHWC:
#     = (1, OFMDim, OFMDim, (ConvKernelDim^2)*IFMChannels)

# note: the actual data layout produced by the hlslib kernels is different
# for depthwise and non-depthwise ops.
# * non-depthwise SWG: (1, OFMDim, OFMDim, K, K, IFMChannels/SIMD, SIMD)
# * depthwise SWG: (1, OFMDim, OFMDim, IFMChannels/SIMD, K, K, SIMD)
# see test_fpgadataflow_slidingwindow.py for an example of how to transform
# between the two layouts


class ConvolutionInputGenerator_MMV(ConvolutionInputGenerator):
    """Class that extends ConvolutionInputGenerator with MMV support during stitching."""

    def __init__(self, onnx_node):
        super().__init__(onnx_node)

    def get_nodeattr_types(self):
        my_attrs = {
            "MMVO" : ("i", False, 1),
            "MMVI" : ("i", False, 1),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_number_output_values(self):
        folded_oshape = self.get_folded_output_shape()
        num_output_elems = np.prod(folded_oshape[:-1])
        return num_output_elems

    def get_exp_cycles(self):
        mmvo = self.get_nodeattr("MMVO")
        return super().get_exp_cycles() // mmvo

    def bram_estimation(self):
        mmvo = self.get_nodeattr("MMVO")
        return mmvo*super().bram_estimation()

    def lut_estimation(self):
        mmvi = self.get_nodeattr("MMVI")
        mmvo = self.get_nodeattr("MMVO")
        simd = self.get_nodeattr("SIMD")
        ifm_ch = self.get_nodeattr("IFMChannels")
        ifm_dim = self.get_nodeattr("IFMDim")
        k = self.get_nodeattr("ConvKernelDim")
        stride = self.get_nodeattr("Stride")
        ram_style = self.get_nodeattr("ram_style")
        if ram_style == "distributed":
            ram_luts = int(
                (k + stride)
                * (
                    mmvi * simd *
                    * self.get_input_datatype().bitwidth()
                    * math.ceil(ifm_dim * ifm_ch / simd / mmvi / 64)
                )
            )
        else:
            ram_luts = 0
        return 300 + ram_luts * mmvo

    def uram_estimation(self):
        mmvo = self.get_nodeattr("MMVO")
        return mmvo*super().uram_estimation()

    def code_generation_ipi(self):
        node_name = self.onnx_node.name
        cmd = []
        ofmdim = self.get_nodeattr("OFMDim")
        ifmdim = self.get_nodeattr("IFMDim")
        ifmch = self.get_nodeattr("IFMChannels")
        simd = self.get_nodeattr("SIMD")
        k = self.get_nodeattr("ConvKernelDim")
        mmvi = self.get_nodeattr("MMVI")
        stride = self.get_nodeattr("Stride")
        precision = self.get_input_datatype().bitwidth()
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
        cmd.append("create_bd_cell -type ip -vlnv user.org:user:swu:1.0 %s/swu" % (node_name))
        padding_height = (ofmdim - (ifmdim - 2))//2
        padding_width = padding_height
        buffer_size =  ifmdim * (ifmch//simd) * k
        if ofmdim//mmvi==0:            
            OFMDIM_MOD_MMV=0
        else:
            OFMDIM_MOD_MMV = 1 
        cmd.append("set_property -dict [list CONFIG.SIMD {%d} \
                                                CONFIG.STRIDE {%d} \
                                                CONFIG.IFMChannels {%d} \
                                                CONFIG.KERNEL_HEIGHT {%d} \
                                                CONFIG.KERNEL_WIDTH {%d} \
                                                CONFIG.IFMWidth {%d} \
                                                CONFIG.IFMHeight {%d} \
                                                CONFIG.PADDING_WIDTH {%d} \
                                                CONFIG.PADDING_HEIGHT {%d} \
                                                CONFIG.OFMWidth {%d} \
                                                CONFIG.OFMHeight {%d} \
                                                CONFIG.IP_PRECISION {%d}\
                                                CONFIG.MMV {%d}\
                                                CONFIG.BUFFER_SIZE {%d}\
                                                CONFIG.OFMDIM_MOD_MMV {%d}] [get_bd_cells %s/swu]" % (simd,
                                                                                        stride,
                                                                                        ifmch,
                                                                                        k,
                                                                                        k,
                                                                                        ifmdim,
                                                                                        ifmdim,
                                                                                        padding_width,
                                                                                        padding_height,
                                                                                        ofmdim,
                                                                                        ofmdim,
                                                                                        precision,
                                                                                        mmvi,
                                                                                        buffer_size,
                                                                                        OFMDIM_MOD_MMV,
                                                                                        node_name
                                                                                        )
                    )
        cmd.append("connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/swu/clk]" % (node_name, clk_name, node_name))
        cmd.append("connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/swu/resetn]" % (node_name, rst_name, node_name))
        cmd.append("connect_bd_intf_net [get_bd_intf_pins %s/%s] [get_bd_intf_pins %s/swu/ip_axis]" % (node_name, din_name, node_name))
        cmd.append("connect_bd_intf_net [get_bd_intf_pins %s/%s] [get_bd_intf_pins %s/swu/op_axis]" % (node_name, dout_name, node_name))
        cmd.append("save_bd_design")
        return cmd
