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

import numpy as np
from onnx import TensorProto, helper
from finn.transformation.base import Transformation
from finn.custom_op.registry import getCustomOp

class MakeFinegrained(Transformation):
    """Convert nodes of a FINN graph to their fine-grained, MMV-capable equivalents"""

    def __init__(self):
        super().__init__()

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        for n in graph.node:
            node_ind += 1
            # convert these layer types:
            # ConvolutionInputGenerator -> ConvolutionInputGenerator_MMV
            #   -absorbs any padding before it
            # StreamingFCLayer_Batch
            #   -emits a Thresholding_Batch if it had thresholding
            # Thresholding_Batch
            if n.op_type == "ConvolutionInputGenerator":
                node_input = n.input[0]
                node_output = n.output[0]
                # pick up existing SWU parameters
                ifmdim = getCustomOp(n).get_nodeattr("IFMDim")
                k = getCustomOp(n).get_nodeattr("ConvKernelDim")
                ich = getCustomOp(n).get_nodeattr("IFMChannels")
                ofmdim = getCustomOp(n).get_nodeattr("OFMDim")
                simd = getCustomOp(n).get_nodeattr("SIMD")
                stride = getCustomOp(n).get_nodeattr("Stride")
                idt = getCustomOp(n).get_nodeattr("inputDataType")
                odt = getCustomOp(n).get_nodeattr("outputDataType")
                dw = getCustomOp(n).get_nodeattr("depthwise")
                ram_style = getCustomOp(n).get_nodeattr("ram_style")
                # defaults for no padding
                padding = [1, 1, 1, 1]
                style = 2
                # see if we have any preceding Padding
                producer = model.find_producer(node_input)
                if producer.op_type == "FMPadding_Batch":
                    node_input = producer.input[0]
                    ifmdim = getCustomOp(producer).get_nodeattr("ImgDim")
                    padding = getCustomOp(producer).get_nodeattr("Padding")
                    style = getCustomOp(producer).get_nodeattr("PaddingStyle")

                new_node = helper.make_node(
                    "ConvolutionInputGenerator_MMV",
                    [node_input],
                    [node_output],
                    domain="finn.custom_op.experimental",
                    backend="fpgadataflow",
                    IFMDim=ifmdim,
                    OFMDim=ofmdim,
                    ConvKernelDim=k,
                    IFMChannels=ich,
                    SIMD=simd,
                    Stride=stride,
                    inputDataType=idt,
                    outputDataType=odt,
                    depthwise=dw,
                    Padding=padding,
                    PaddingStyle=style,
                    MMVI=1,
                    MMVO=1,
                )
                graph.node.insert(node_ind, new_node)
                # remove old nodes
                graph.node.remove(n)
                if producer.op_type == "FMPadding_Batch":
                    graph.node.remove(producer)
                return (model, True)

            elif n.op_type == "StreamingFCLayer_Batch" or n.op_type == "Vector_Vector_Activate_Batch":
                is_vvau = (n.op_type == "Vector_Vector_Activate_Batch")
                node_input = n.input[0]
                weight_input = n.input[1]
                node_output = n.output[0]
                # if runtime-writable weights, can't use finegrained
                if getCustomOp(n).get_nodeattr("runtime_writeable_weights") == 1:
                    continue
                #TODO: decouple the activation if needed
                assert getCustomOp(n).get_nodeattr("noActivation") == 1
                # copy all relevant parameters to new node
                mmode = getCustomOp(n).get_nodeattr("mem_mode")
                if mmode == "const":
                    mmode = "decoupled"
                new_node = helper.make_node(
                    "StreamingFCLayer_MMV_FG_Batch",
                    [node_input, weight_input],
                    [node_output],
                    domain="finn.custom_op.experimental",
                    backend="fpgadataflow",
                    PE = getCustomOp(n).get_nodeattr("PE"),
                    SIMD = 1 if is_vvau else getCustomOp(n).get_nodeattr("SIMD"),
                    MW = getCustomOp(n).get_nodeattr("MW"),
                    MH = getCustomOp(n).get_nodeattr("MH"),
                    resType = getCustomOp(n).get_nodeattr("resType"),
                    actVal = getCustomOp(n).get_nodeattr("ActVal"),
                    inputDataType = getCustomOp(n).get_nodeattr("inputDataType"),
                    weightDataType = getCustomOp(n).get_nodeattr("weightDataType"),
                    outputDataType = getCustomOp(n).get_nodeattr("outputDataType"),
                    accDataType = getCustomOp(n).get_nodeattr("accDataType"),
                    binaryXnorMode = getCustomOp(n).get_nodeattr("binaryXnorMode"),
                    numInputVectors = getCustomOp(n).get_nodeattr("numInputVectors"),
                    mem_mode = mmode,
                    ram_style = getCustomOp(n).get_nodeattr("ram_style"),
                    ibuf_ram_style = getCustomOp(n).get_nodeattr("ram_style"),
                    MMV = 1,
                    VVAU = 1 if is_vvau else 0,
                )
                graph.node.insert(node_ind, new_node)
                # remove old nodes
                graph.node.remove(n)
                return (model, True)
        return (model, False)
