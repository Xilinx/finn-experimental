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
from finn.util.ipi_axis_stitch import (
    axis_gather_bcast_scatter,
    axis_buffer,
    axis_pack,
    axis_unpack,
)

class ActivationTransport_MMV_Batch(StreamingDataWidthConverter_Batch):
    """Class that extends StreamingDataWidthConverter_Batch with MMV support."""
    #TODO: implement MMV initiation and termination

    def __init__(self, onnx_node):
        super().__init__(onnx_node)

    def get_nodeattr_types(self):
        my_attrs = {
            "MMV" : ("i", False, 1),
            "IFIFODepth": ("i", False, 0),
            "OFIFODepth": ("i", False, 0),
            "IFIFORamStyle": ("s", False, "auto", {"auto", "block", "distributed", "ultra"}),
            "OFIFORamStyle": ("s", False, "auto", {"auto", "block", "distributed", "ultra"}),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def code_generation_ipi(self):
        mmv_value = self.get_nodeattr("MMV")
        ifd = self.get_nodeattr("IFIFODepth")
        ofd = self.get_nodeattr("OFIFODepth")
        ifrs = self.get_nodeattr("IFIFORamStyle")
        ofrs = self.get_nodeattr("OFIFORamStyle")
        node_name = self.onnx_node.name
        ibits = self.get_instream_width()
        obits = self.get_outstream_width()
        assert (ibits % obits == 0 or obits % ibits == 0)

        clk_name = self.get_verilog_top_module_intf_names()["clk"][0]
        rst_name = self.get_verilog_top_module_intf_names()["rst"][0]
        dout_name = self.get_verilog_top_module_intf_names()["m_axis"][0][0]
        din_name = self.get_verilog_top_module_intf_names()["s_axis"][0][0]

        cmd = []
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

        # input FIFO
        cmd += axis_buffer("ififo", ifd, mmv_value*ibits, ifrs, parent_hier=node_name)
        cmd.append(
            "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/ififo/aresetn]"
            % (node_name, rst_name, node_name)
        )
        cmd.append(
            "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/ififo/aclk]"
            % (node_name, clk_name, node_name)
        )
        cmd.append(
            "connect_bd_intf_net [get_bd_intf_pins %s/%s] [get_bd_intf_pins %s/ififo/s_axis]"
            % (node_name, din_name, node_name)
        )

        # output FIFO
        cmd += axis_buffer("ofifo", ofd, mmv_value*obits, ofrs, parent_hier=node_name)
        cmd.append(
            "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/ofifo/aresetn]"
            % (node_name, rst_name, node_name)
        )
        cmd.append(
            "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/ofifo/aclk]"
            % (node_name, clk_name, node_name)
        )
        cmd.append(
            "connect_bd_intf_net [get_bd_intf_pins %s/%s] [get_bd_intf_pins %s/ofifo/m_axis]"
            % (node_name, dout_name, node_name)
        )

        if ibits == obits:
            #no width conversion necessary, just connect fifos together and return
            cmd.append(
                "connect_bd_intf_net [get_bd_intf_pins %s/ififo/m_axis] [get_bd_intf_pins %s/ofifo/s_axis]"
                % (node_name, node_name)
            )
            return cmd

        # split MMV
        cmd += axis_gather_bcast_scatter("mmv_split", 1, 1, mmv_value, ibits*mmv_value, parent_hier=node_name)
        cmd.append(
            "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/mmv_split/aresetn]"
            % (node_name, rst_name, node_name)
        )
        cmd.append(
            "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/mmv_split/aclk]"
            % (node_name, clk_name, node_name)
        )
        cmd.append(
            "connect_bd_intf_net [get_bd_intf_pins %s/ififo/m_axis] [get_bd_intf_pins %s/mmv_split/s_0_axis]"
            % (node_name, node_name)
        )

        # join MMV
        cmd += axis_gather_bcast_scatter("mmv_join", mmv_value, 1, 1, obits, parent_hier=node_name)
        cmd.append(
            "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/mmv_join/aresetn]"
            % (node_name, rst_name, node_name)
        )
        cmd.append(
            "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/mmv_join/aclk]"
            % (node_name, clk_name, node_name)
        )
        cmd.append(
            "connect_bd_intf_net [get_bd_intf_pins %s/ofifo/s_axis] [get_bd_intf_pins %s/mmv_join/m_0_0_axis]"
            % (node_name, node_name)
        )

        # find element size, which is the greatest common divisor of ibits and obits
        # this should be equal to either ibits or obits TODO: assert for this
        elem_size = math.gcd(ibits, obits)
        # DWC input takes 
        dwc_ibytes = (ibits//elem_size)*int(math.ceil(elem_size/8))
        dwc_obytes = (obits//elem_size)*int(math.ceil(elem_size/8))

        # for every MMV lane, do conversion
        for m in range(mmv_value):
            #if up-converting, dwc then pack
            #if down-converting, unpack then dwc

            #convert
            cmd.append(
                "create_bd_cell -type ip "
                "-vlnv xilinx.com:ip:axis_dwidth_converter:1.1 /%s/dwc_%d" % (node_name, m)
            )
            cmd.append(
                "set_property -dict "
                "[list CONFIG.S_TDATA_NUM_BYTES.VALUE_SRC USER] "
                "[get_bd_cells /%s/dwc_%d]" % (node_name, m)
            )
            cmd.append(
                "set_property -dict "
                "[list CONFIG.S_TDATA_NUM_BYTES {%d}] [get_bd_cells /%s/dwc_%d]"
                % (dwc_ibytes, node_name, m)
            )
            cmd.append(
                "set_property -dict "
                "[list CONFIG.M_TDATA_NUM_BYTES {%d}] [get_bd_cells /%s/dwc_%d]"
                % (dwc_obytes, node_name, m)
            )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/dwc_%d/aresetn]"
                % (node_name, rst_name, node_name, m)
            )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/dwc_%d/aclk]"
                % (node_name, clk_name, node_name, m)
            )

            #unpack 
            if ibits > obits:
                cmd += axis_unpack("unpack_"+str(m), int(ibits/elem_size), elem_size, parent_hier=node_name)
                cmd.append(
                    "connect_bd_intf_net [get_bd_intf_pins %s/mmv_split/m_0_%d_axis] [get_bd_intf_pins %s/unpack_%d/s_axis]"
                    % (node_name, m, node_name, m)
                )
                cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/unpack_%d/aresetn]"
                    % (node_name, rst_name, node_name, m)
                )
                cmd.append(
                    "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/unpack_%d/aclk]"
                    % (node_name, clk_name, node_name, m)
                )
                cmd.append(
                    "connect_bd_intf_net [get_bd_intf_pins %s/dwc_%d/S_AXIS] [get_bd_intf_pins %s/unpack_%d/m_axis]"
                    % (node_name, m, node_name, m)
                )
            else:
                cmd.append(
                    "connect_bd_intf_net [get_bd_intf_pins %s/mmv_split/m_0_%d_axis] [get_bd_intf_pins %s/dwc_%d/S_AXIS]"
                    % (node_name, m, node_name, m)
                )

            #repack
            if ibits < obits:
                cmd += axis_pack("pack_"+str(m), int(obits/elem_size), elem_size, parent_hier=node_name)
                cmd.append(
                    "connect_bd_intf_net [get_bd_intf_pins %s/mmv_join/s_%d_axis] [get_bd_intf_pins %s/pack_%d/m_axis]"
                    % (node_name, m, node_name, m)
                )
                cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/pack_%d/aresetn]"
                    % (node_name, rst_name, node_name, m)
                )
                cmd.append(
                    "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/pack_%d/aclk]"
                    % (node_name, clk_name, node_name, m)
                )
                cmd.append(
                    "connect_bd_intf_net [get_bd_intf_pins %s/dwc_%d/M_AXIS] [get_bd_intf_pins %s/pack_%d/s_axis]"
                    % (node_name, m, node_name, m)
                )
            else:
                cmd.append(
                    "connect_bd_intf_net [get_bd_intf_pins %s/mmv_join/s_%d_axis] [get_bd_intf_pins %s/dwc_%d/M_AXIS]"
                    % (node_name, m, node_name, m)
                )

        return cmd

    def lut_estimation(self):
        """Calculates resource estimations for LUTs"""
        mmv_value = self.get_nodeattr("MMV")
        return mmv_value*super().lut_estimation()

    def code_generation_ipgen(self, model, fpgapart, clk):
        pass

    def ipgen_singlenode_code(self):
        self.set_nodeattr("ipgen_path", self.get_nodeattr("code_gen_dir_ipgen"))
        self.set_nodeattr("ip_path", self.get_nodeattr("code_gen_dir_ipgen"))
