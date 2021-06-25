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

import math
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
       
        for node in graph.node:
            node_ind += 1
            # convert these layer types:
            # ConvolutionInputGenerator -> ConvolutionInputGenerator_MMV
            #   -absorbs any padding before it
            # StreamingFCLayer_Batch
            #   -emits a Thresholding_Batch if it had thresholding
            # Thresholding_Batch
            if node.op_type == "FMPadding_Batch":
                #iteratively remove everything between a padding node and the subsequent SWU
                node_output = node.output[0]
                consumer = model.find_producer(node_output)
                if consumer is not None:
                    if producer.op_type != "ConvolutionInputGenerator":
                        node.output[0] = consumer.output[0]
                        graph.node.remove(consumer)
                        return (model, True)

            if node.op_type == "ConvolutionInputGenerator":
                node_input = node.input[0]
                node_output = node.output[0]
                # pick up existing SWU parameters
                ifmdim = getCustomOp(node).get_nodeattr("IFMDim")
                k = getCustomOp(node).get_nodeattr("ConvKernelDim")
                ich = getCustomOp(node).get_nodeattr("IFMChannels")
                ofmdim = getCustomOp(node).get_nodeattr("OFMDim")
                simd = getCustomOp(node).get_nodeattr("SIMD")
                stride = getCustomOp(node).get_nodeattr("Stride")
                idt = getCustomOp(node).get_nodeattr("inputDataType")
                odt = getCustomOp(node).get_nodeattr("outputDataType")
                dw = getCustomOp(node).get_nodeattr("depthwise")
                ram_style = getCustomOp(node).get_nodeattr("ram_style")
                # defaults for no padding
                padding = [0,0,0,0]
                style = 2
                # see if we have any preceding Padding
                producer = model.find_producer(node_input)
                if producer is not None:
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
                graph.node.remove(node)
                if producer is not None:
                    if producer.op_type == "FMPadding_Batch":
                        graph.node.remove(producer)
                return (model, True)

            elif node.op_type == "StreamingFCLayer_Batch" or node.op_type == "Vector_Vector_Activate_Batch":
                is_vvau = (node.op_type == "Vector_Vector_Activate_Batch")
                node_input = node.input[0]
                weight_input = node.input[1]
                node_output = node.output[0]
                # if MVAU with runtime-writable weights, can't use finegrained
                if not is_vvau:
                    if getCustomOp(node).get_nodeattr("runtime_writeable_weights") == 1:
                        continue
                #TODO: decouple the activation if needed
                assert getCustomOp(node).get_nodeattr("noActivation") == 1
                # copy all relevant parameters to new node
                # mem mode for MVAU can be const, decoupled, external
                # we can handle decoupled and external
                # turn const to decoupled, keep external
                if not is_vvau:
                    mmode = getCustomOp(node).get_nodeattr("mem_mode")
                    if mmode == "const":
                        mmode = "decoupled"
                new_node = helper.make_node(
                    "StreamingFCLayer_MMV_FG_Batch",
                    [node_input, weight_input],
                    [node_output],
                    domain="finn.custom_op.experimental",
                    backend="fpgadataflow",
                    PE = getCustomOp(node).get_nodeattr("PE"),
                    SIMD = 1 if is_vvau else getCustomOp(node).get_nodeattr("SIMD"),
                    MW = -1 if is_vvau else getCustomOp(node).get_nodeattr("MW"),
                    MH = -1 if is_vvau else getCustomOp(node).get_nodeattr("MH"),
                    resType = getCustomOp(node).get_nodeattr("resType"),
                    actVal = getCustomOp(node).get_nodeattr("ActVal"),
                    inputDataType = getCustomOp(node).get_nodeattr("inputDataType"),
                    weightDataType = getCustomOp(node).get_nodeattr("weightDataType"),
                    outputDataType = getCustomOp(node).get_nodeattr("outputDataType"),
                    accDataType = getCustomOp(node).get_nodeattr("accDataType"),
                    binaryXnorMode = 0 if is_vvau else getCustomOp(node).get_nodeattr("binaryXnorMode"),
                    numInputVectors = -1 if is_vvau else getCustomOp(node).get_nodeattr("numInputVectors"),
                    mem_mode = "decoupled" if is_vvau else mmode,
                    ram_style = "auto" if is_vvau else getCustomOp(node).get_nodeattr("ram_style"),
                    ibuf_ram_style = "auto" if is_vvau else getCustomOp(node).get_nodeattr("ram_style"),
                    MMV = 1,
                    VVAU = 1 if is_vvau else 0,
                )
                graph.node.insert(node_ind, new_node)
                # remove old nodes
                graph.node.remove(node)
                return (model, True)

            elif node.op_type == "Thresholding_Batch":
                if getCustomOp(node).get_nodeattr("runtime_writeable_weights") == 1:
                    continue
                node_input = node.input[0]
                weight_input = node.input[1]
                node_output = node.output[0]
                new_node = helper.make_node(
                    "Thresholding_MMV_Batch",
                    [node_input, weight_input],
                    [node_output],
                    domain="finn.custom_op.experimental",
                    backend="fpgadataflow",
                    NumChannels = getCustomOp(node).get_nodeattr("NumChannels"),
                    PE = getCustomOp(node).get_nodeattr("PE"),
                    numSteps = getCustomOp(node).get_nodeattr("numSteps"),
                    ActVal = getCustomOp(node).get_nodeattr("ActVal"),
                    inputDataType = getCustomOp(node).get_nodeattr("inputDataType"),
                    weightDataType = getCustomOp(node).get_nodeattr("weightDataType"),
                    outputDataType = getCustomOp(node).get_nodeattr("outputDataType"),
                    numInputVectors = getCustomOp(node).get_nodeattr("numInputVectors"),
                    mem_mode = "decoupled",
                    ram_style = getCustomOp(node).get_nodeattr("ram_style"),
                    MMV = 1,
                    runtime_writeable_weights = 0,
                )
                graph.node.insert(node_ind, new_node)
                # remove old nodes
                graph.node.remove(node)
                return (model, True)

            elif node.op_type == "StreamingDataWidthConverter_Batch":
                node_input = node.input[0]
                node_output = node.output[0]
                depth = 0
                style = "auto"
                # watch out for DWC -> FIFO pattern
                consumer = model.find_consumer(node_output)
                if consumer is not None:
                    if consumer.op_type == "StreamingFIFO":
                        node_output = consumer.output[0]
                        depth = getCustomOp(consumer).get_nodeattr("depth")
                        style = getCustomOp(consumer).get_nodeattr("ram_style")
                new_node = helper.make_node(
                    "ActivationTransport_MMV_Batch",
                    [node_input],
                    [node_output],
                    domain="finn.custom_op.experimental",
                    backend="fpgadataflow",
                    inWidth=getCustomOp(node).get_nodeattr("inWidth"),
                    outWidth=getCustomOp(node).get_nodeattr("outWidth"),
                    shape=getCustomOp(node).get_nodeattr("shape"),
                    dataType=getCustomOp(node).get_nodeattr("dataType"),
                    MMV = 1,
                    IFIFODepth = 0,
                    OFIFODepth = depth,
                    OFIFORamStyle = style,
                )
                graph.node.insert(node_ind, new_node)
                # remove old nodes
                graph.node.remove(node)
                if consumer is not None:
                    if consumer.op_type == "StreamingFIFO":
                        graph.node.remove(consumer)
                return (model, True)

            elif node.op_type == "StreamingFIFO":
                node_input = node.input[0]
                node_output = node.output[0]
                iw = getCustomOp(node).get_instream_width()
                ow = iw
                depth = getCustomOp(node).get_nodeattr("depth")
                style = getCustomOp(node).get_nodeattr("ram_style")
                dt = getCustomOp(node).get_nodeattr("dataType")
                shape=getCustomOp(node).get_nodeattr("folded_shape")
                consumer = model.find_consumer(node_output)
                consumer2 = None
                if consumer is not None:
                    if consumer.op_type == "StreamingDataWidthConverter_Batch":
                        # FIFO -> DWC pattern
                        node_output = consumer.output[0]
                        ow = getCustomOp(consumer).get_nodeattr("outWidth")
                        shape=getCustomOp(consumer).get_nodeattr("shape")
                        consumer2 = model.find_consumer(node_output)
                        if consumer2 is not None:
                            if consumer2.op_type == "StreamingFIFO":
                                # FIFO -> DWC -> FIFO pattern
                                depth += int(math.ceil((ow/iw)*getCustomOp(consumer2).get_nodeattr("depth")))
                                node_output = consumer2.output[0]
                new_node = helper.make_node(
                    "ActivationTransport_MMV_Batch",
                    [node_input],
                    [node_output],
                    domain="finn.custom_op.experimental",
                    backend="fpgadataflow",
                    shape=shape,
                    dataType=dt,
                    inWidth=iw,
                    outWidth=ow,
                    MMV = 1,
                    OFIFODepth = 0,
                    IFIFODepth = depth,
                    IFIFORamStyle = style,
                )
                graph.node.insert(node_ind, new_node)
                # remove old nodes
                graph.node.remove(node)
                if consumer is not None:
                    if consumer.op_type == "StreamingDataWidthConverter_Batch":
                        graph.node.remove(consumer)
                if consumer2 is not None:
                    if consumer2.op_type == "StreamingFIFO_Batch":
                        graph.node.remove(consumer2)
                return (model, True)
                
        return (model, False)
