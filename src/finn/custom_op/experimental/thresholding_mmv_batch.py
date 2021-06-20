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
from finn.custom_op.fpgadataflow import templates

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

        mmv = self.get_nodeattr("MMV")
        if mmv == 1:
           return super().code_generation_ipi()

        if mem_mode == "decoupled":
            node_name = self.onnx_node.name
            # create a hierarchy for this layer, with the same port names
            clk_name = self.get_verilog_top_module_intf_names()["clk"][0]
            rst_name = self.get_verilog_top_module_intf_names()["rst"][0]
            dout_name = self.get_verilog_top_module_intf_names()["m_axis"][0]
            din_name = self.get_verilog_top_module_intf_names()["s_axis"][0]
            weight_name = self.get_verilog_top_module_intf_names()["s_axis"][1]
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

            #connect streamer clk, reset
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/aclk]"
                % (node_name, clk_name, node_name, strm_inst)
            )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/aresetn]"
                % (node_name, rst_name, node_name, strm_inst)
            )

            # instantiate input scatter
            iwidth = int(math.ceil(self.get_instream_width()/8))*8 * mmv
            owidth = int(math.ceil(self.get_outstream_width()/8))*8 * mmv
            cmd += axis_gather_bcast_scatter("immv_transport", 1, 1, mmv, (iwidth//8), parent_hier=node_name)
            #connect it to input/clk/rst
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/immv_transport/aclk]"
                % (node_name, clk_name, node_name)
            )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/immv_transport/aresetn]"
                % (node_name, rst_name, node_name)
            )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/immv_transport/s_0_axis]"
                % (node_name, din_name, node_name)
            )

            # instantiate weight broadcast
            wwidth=self.get_weightstream_width()
            wwidth_padded=roundup_to_integer_multiple(wwidth, 8)

            # weight transport subhierarchy
            cmd += axis_gather_bcast_scatter("weight_transport", 1, mmv, 1, wwidth_padded//8, parent_hier=node_name)
            #connect it to input/clk/rst
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/weight_transport/aclk]"
                % (node_name, clk_name, node_name)
            )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/weight_transport/aresetn]"
                % (node_name, rst_name, node_name)
            )

            # connect output of streamer to input of weight broadcaster
            cmd.append("connect_bd_intf_net [get_bd_intf_pins %s/%s/m_axis_0] "
                        "[get_bd_intf_pins %s/weight_transport/s_0_axis]"
                        % (node_name, strm_inst, node_name)
            )

            # instantiate output gather
            cmd += axis_gather_bcast_scatter("out_transport", mmv, 1, 1, owidth, parent_hier=node_name)
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/out_transport/aclk]"
                % (node_name, clk_name, node_name)
            )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/out_transport/aresetn]"
                % (node_name, rst_name, node_name)
            )
            cmd.append(
                "connect_bd_intf_net [get_bd_intf_pins %s/out_transport/m_0_0_axis] "
                "[get_bd_intf_pins %s/%s]"
                % (node_name, node_name, dout_name)
            )

            # instantiate and connect HLS IPs
            for i in range(mmv):
                cell_name = "thr_"+str(i)
                cmd.append(
                    "create_bd_cell -type ip -vlnv %s /%s/%s"
                    % (self.get_nodeattr("ip_vlnv"), node_name, cell_name)
                )
                cmd.append(
                    "connect_bd_intf_net [get_bd_intf_pins %s/out_transport/s_%d_axis] "
                    "[get_bd_intf_pins %s/%s/%s]"
                    % (node_name, i, node_name, cell_name, dout_name)
                )
                cmd.append(
                    "connect_bd_intf_net [get_bd_intf_pins %s/immv_transport/m_%d_0_axis] "
                    "[get_bd_intf_pins %s/%s/%s]"
                    % (node_name, i, node_name, cell_name, din_name)
                )
                cmd.append(
                    "connect_bd_intf_net [get_bd_intf_pins %s/weight_transport/m_%d_0_axis] "
                    "[get_bd_intf_pins %s/%s/%s]"
                    % (node_name, i, node_name, cell_name, weight_name)
                )

            cmd.append("save_bd_design")
        else:
            raise Exception("Unrecognized mem_mode for Thresholding_MMV_Batch")

        return cmd

