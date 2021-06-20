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
from finn.custom_op.fpgadataflow.hlscustomop import HLSCustomOp
from finn.custom_op.general.im2col import compute_conv_output_dim
from onnx import TensorProto, helper
from finn.util.basic import (
    roundup_to_integer_multiple,
)

class ConvolutionInputGenerator_MMV(HLSCustomOp):
    """Class that extends ConvolutionInputGenerator with MMV support during stitching."""

    def __init__(self, onnx_node):
        super().__init__(onnx_node)

    def get_nodeattr_types(self):
        my_attrs = {
            "ConvKernelDim": ("ints", True, []),  # [H, W] = [Y, X]
            "IFMChannels": ("i", True, 0),
            "IFMDim": ("ints", True, []),  # [H, W] = [Y, X]
            "OFMDim": ("ints", True, []),  # [H, W] = [Y, X]
            "SIMD": ("i", True, 0),
            "Stride": ("ints", True, [1, 1]),  # [H, W] = [Y, X]
            # note: only dilation=1 supported for now
            "Dilation": ("ints", True, [1, 1]),  # [H, W] = [Y, X]
            # FINN DataTypes for inputs, weights, outputs
            "inputDataType": ("s", True, ""),
            "outputDataType": ("s", True, ""),
            "depthwise": ("i", False, 0, {0, 1}),
            "MMVO" : ("i", False, 1),
            "MMVI" : ("i", False, 1),
            # total padding (per dimension) to apply
            # NOTE: Current padding scheme that is applied tries to pad the same
            # amount of zeros in front and behind the image for each dimension.
            # As an example, a padding scheme such as [1, x, 3, x] is equal
            # to [2, x, 2, x]
            "Padding": (
                "ints",
                True,
                [1, 1, 1, 1],
            ),  # [H_begin, W_begin, H_end, W_end] = [Y_begin, X_begin, Y_end, X_end]
            # controls distribution of padded pixels
            # in case of uneven padding -- see FMPadding fxn
            # in hlslib
            "PaddingStyle": ("i", False, 2, {2, 1}),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_instream_width(self):
        """Returns stream width, input and output stream width are equal for
        the sliding window function"""
        ibits = self.get_input_datatype().bitwidth()
        simd = self.get_nodeattr("SIMD")
        immv = self.get_nodeattr("MMVI")
        in_width = simd * ibits * immv
        return in_width

    def get_outstream_width(self):
        """Returns stream width, input and output stream width are equal for
        the sliding window function, so the function to determine the input
        stream width can be reused."""
        ibits = self.get_input_datatype().bitwidth()
        simd = self.get_nodeattr("SIMD")
        ommv = self.get_nodeattr("MMVO")
        out_width = simd * ibits * ommv
        return out_width

    def get_input_datatype(self):
        """Returns FINN DataType of input."""
        return DataType[self.get_nodeattr("inputDataType")]

    def get_output_datatype(self):
        """Returns FINN DataType of output."""
        return DataType[self.get_nodeattr("outputDataType")]

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
                    mmvi * simd
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

    def get_verilog_top_module_intf_names(self):
        intf_names = {}
        intf_names["clk"] = ["aclk"]
        intf_names["rst"] = ["aresetn"]
        intf_names["s_axis"] = [("s_axis", self.get_instream_width_padded())]
        intf_names["m_axis"] = [("m_axis", self.get_outstream_width_padded())]
        intf_names["aximm"] = []
        intf_names["axilite"] = []
        return intf_names

    def code_generation_ipi(self):
        node_name = self.onnx_node.name
        cmd = []
        ofmdim = self.get_nodeattr("OFMDim")
        ifmdim = self.get_nodeattr("IFMDim")
        ifmch = self.get_nodeattr("IFMChannels")
        simd = self.get_nodeattr("SIMD")
        k = self.get_nodeattr("ConvKernelDim")
        mmvi = self.get_nodeattr("MMVI")
        mmvo = self.get_nodeattr("MMVO")
        stride = self.get_nodeattr("Stride")
        precision = self.get_input_datatype().bitwidth()
        # create a hierarchy for this layer, with the same port names
        clk_name = self.get_verilog_top_module_intf_names()["clk"][0]
        rst_name = self.get_verilog_top_module_intf_names()["rst"][0]
        dout_name = self.get_verilog_top_module_intf_names()["m_axis"][0][0]
        din_name = self.get_verilog_top_module_intf_names()["s_axis"][0][0]
        cmd.append("create_bd_cell -type ip -vlnv xilinx.com:user:swu:1.0 %s" % (node_name))
        padding_height = (ofmdim[1] - (ifmdim[1] - 2))//2
        padding_width = (ofmdim[0] - (ifmdim[0] - 2))//2
        buffer_size =  ifmdim[0] * (ifmch//simd) * k[1]
        if ofmdim[0]//mmvi==0:            
            OFMDIM_MOD_MMV=0
        else:
            OFMDIM_MOD_MMV = 1 
        cmd.append("set_property -dict [list CONFIG.SIMD {%d} \
                                        CONFIG.STRIDE_HT {%d} \
                                        CONFIG.STRIDE_WT {%d} \
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
                                        CONFIG.MMV_IN {%d}\
                                        CONFIG.MMV_OUT {%d}]\
                                        [get_bd_cells %s]" % (simd,
                                                            stride[0],
                                                            stride[1],
                                                            ifmch,
                                                            k[1],
                                                            k[0],
                                                            ifmdim[0],
                                                            ifmdim[1],
                                                            padding_width,
                                                            padding_height,
                                                            ofmdim[0],
                                                            ofmdim[1],
                                                            precision,
                                                            mmvi,
                                                            mmvo,
                                                            node_name
                                                            )
                    )
        cmd.append("save_bd_design")
        return cmd


    def blackboxfunction(self):
        pass
    
    def dataoutstrm(self):
        pass
    
    def defines(self):
        pass
    
    def docompute(self):
        pass
    
    def global_includes(self):
        pass
    
    def infer_node_datatype(self):
        pass
    
    def make_shape_compatible_op(self):
        pass
    
    def pragmas(self):
        pass
    
    def read_npy_data(self):
        pass
    
    def save_as_npy(self):
        pass
    
    def strm_decl(self):
        pass
    
    def verify_node(self):
        pass

    def get_number_output_values(self):
        pass

    def code_generation_ipgen(self, model, fpgapart, clk):
        pass

    def ipgen_singlenode_code(self):
        self.set_nodeattr("ipgen_path", self.get_nodeattr("code_gen_dir_ipgen"))
        self.set_nodeattr("ip_path", self.get_nodeattr("code_gen_dir_ipgen"))