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
from qonnx.core.datatype import DataType
from onnx import TensorProto, helper

from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import get_by_name


class InferDoublePackedConv(Transformation):
    """InferDoublePackedConv"""

    def __init__(self, pos=None):
        super(InferDoublePackedConv, self).__init__()
        # pos can be an iterable of integers representing positions or None
        # if pos is None, we'll insert doublepacked Convs everywhere
        # else we insert in the positions specified
        self.conv_position_to_replace = None
        if pos is not None:
            self.conv_position_to_replace = tuple(pos)

    def get_smallest_possible(self, vals):
        """Returns smallest (fewest bits) possible DataType that can represent
        value. Prefers unsigned integers where possible."""
        vals = np.array(vals)
        for v in vals:
            assert int(v) == v, "Error float value"

        cands = DataType.get_accumulator_dt_cands()
        for k in cands:
            dt = DataType[k]
            if (dt.min() <= vals).all() and (vals <= dt.max()).all():
                return dt

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        conv_position = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Conv":
                conv_position += 1
                if self.conv_position_to_replace is not None:
                    if conv_position not in self.conv_position_to_replace:
                        continue

                cnv_input = n.input[0]
                cnv_output = n.output[0]
                idt = model.get_tensor_datatype(cnv_input)
                odt = model.get_tensor_datatype(cnv_output)
                # extract conv parameters
                k = get_by_name(n.attribute, "kernel_shape").ints[-1]
                pad = get_by_name(n.attribute, "pads").ints[-1]
                stride = get_by_name(n.attribute, "strides").ints[-1]
                weight_name = n.input[1]
                W_conv = model.get_initializer(weight_name)
                ifm_ch = W_conv.shape[1]
                ofm_ch = W_conv.shape[0]
                ifm_dim = model.get_tensor_shape(n.input[0])[-1]  # assume NCHW
                ofm_dim = model.get_tensor_shape(n.output[0])[-1]  # assume NCHW
                # reuse conv weights for new matmul weights
                # conv weights are [OFM][IFM][k][k]
                # first convert to [OFM][k][k][IFM] (to remain compatible with
                # finn-hlslib and how it does im2col/sliding window)
                W_matmul = W_conv.transpose(0, 2, 3, 1)
                # reshape into [OFM][k*k*IFM] matrix

                mh = ofm_ch
                mw = ifm_ch * k * k
                W_matmul = W_matmul.reshape(mh, mw)
                # transpose to get ONNX-compatible [k*k*IFM][OFM] matrix
                W_matmul = W_matmul.T

                model.set_initializer(weight_name, W_matmul)
                wdt = self.get_smallest_possible(
                    [min(W_matmul.flatten()), max(W_matmul.flatten())]
                )

                if wdt.bitwidth() > 8:
                    print(
                        "Can't infer double packed conv as weight bits =",
                        wdt.bitwidth(),
                    )
                    continue
                if wdt.signed():
                    wdt = DataType["INT8"]
                else:
                    wdt = DataType["UINT8"]

                model.set_tensor_datatype(weight_name, wdt)
                idtypes = [idt, wdt]
                has_signed_inp = len(list(filter(lambda x: x.signed(), idtypes))) != 0
                if has_signed_inp:
                    acc_dt = DataType["INT32"]
                else:
                    acc_dt = DataType["UINT32"]

                # create new intermediate values
                inp_trans_out = helper.make_tensor_value_info(
                    model.make_new_valueinfo_name(),
                    TensorProto.FLOAT,
                    (1, ifm_dim, ifm_dim, ifm_ch),  # NHWC
                )
                graph.value_info.append(inp_trans_out)
                inp_trans_out = inp_trans_out.name
                model.set_tensor_datatype(inp_trans_out, idt)

                # create new nodes
                # NCHW -> NHWC
                inp_trans_node = helper.make_node(
                    "Transpose", [cnv_input], [inp_trans_out], perm=[0, 2, 3, 1]
                )

                conv_node_inputs = [inp_trans_out, weight_name]

                dp_node_out = helper.make_tensor_value_info(
                    model.make_new_valueinfo_name(),
                    TensorProto.FLOAT,
                    (1, ofm_dim, ofm_dim, ofm_ch),
                )
                graph.value_info.append(dp_node_out)
                dp_node_out = dp_node_out.name

                has_activation = False
                consumer = model.find_consumer(cnv_output)
                if consumer is not None and consumer.op_type == "MultiThreshold":
                    has_activation = True

                    mt_output = consumer.output[0]
                    mt_thres = consumer.input[1]
                    T = model.get_initializer(mt_thres)
                    # ensure integer thresholds?

                    Tnew = np.ceil(T)
                    if acc_dt.is_integer() and (T != Tnew).any():
                        # round up the thresholds to nearest integer
                        model.set_initializer(mt_thres, Tnew)
                        # use same datatype as inputs for thresholds
                        model.set_tensor_datatype(mt_thres, acc_dt)
                    if acc_dt.is_integer() and not acc_dt.signed() and (Tnew < 0).any():
                        # clip any negative thresholds
                        Tnew = np.clip(Tnew, 0, None)
                        model.set_initializer(mt_thres, Tnew)
                        # use same datatype as inputs for thresholds
                        model.set_tensor_datatype(mt_thres, acc_dt)

                    # create MVTU (i.e. including activation)

                    assert (
                        T.shape[0] == 1 or T.shape[0] == mh
                    ), """First dimension of
                    thresholds neither 1 nor MH."""
                    scale = getCustomOp(consumer).get_nodeattr("out_scale")
                    assert (
                        scale == 1.0
                    ), "out_scale must be equal to 1.0 for HLS conversion."
                    actval = getCustomOp(consumer).get_nodeattr("out_bias")
                    assert (
                        0 == actval
                    ), "out_bias must be 0 for HLS conversion of Dpacked conv."

                    # model.set_tensor_shape(mm_input, mm_in_shape)
                    # model.set_tensor_shape(mt_output, mt_out_shape)

                    out_transp_out = mt_output
                    conv_node_inputs += [mt_thres]
                    odt = model.get_tensor_datatype(mt_output)
                else:
                    out_transp_out = cnv_output
                    odt = acc_dt

                model.set_tensor_datatype(dp_node_out, odt)

                # dp conv
                simd = 1
                pe = 1
                assert mh % pe == 0, "Requirement MH divisible by PE is violated."
                assert mw % simd == 0, "Requirement MW divisible by SIMD is violated."
                wmem = mw * mh // (pe * simd)
                assert (
                    mw * mh == wmem * pe * simd
                ), "Requirement (MW * MH) divisiable by(WMEM * PE * SIMD) is violated."

                dp_conv_node = helper.make_node(
                    "ConvDoublePacked_Batch",
                    conv_node_inputs,
                    [dp_node_out],
                    domain="finnexperimental.custom_op.experimental",
                    backend="fpgadataflow",
                    ConvKernelDim=k,  # ("i",True,0),
                    IFMChannels=ifm_ch,  # ("i",True,0),
                    IFMDim=ifm_dim,  # ("i",True,0),
                    OFMChannels=ofm_ch,  # ("i",True,0),
                    OFMDim=ofm_dim,  # ("i", True, 0),
                    Stride=stride,  # ("i",True,0),
                    Padding=pad,  # ("i",True,0),
                    SIMD=simd,  # ("i",True,0),
                    PE=pe,  # ("i",True,0),           #num
                    MW=mw,  # ("i", True, 0),
                    MH=mh,  # ("i", True, 0),
                    inputDataType=idt.name,  # ("s", True, ""),
                    weightDataType=wdt.name,  # ("s", True, ""),
                    outputDataType=odt.name,  # ("s", True, ""),
                    noActivation=0
                    if has_activation
                    else 1,  # ("i", False, 0), #"ActivationType ("i",True,0)
                    numInputVectors=[1, ofm_dim, ofm_dim],  # ("ints", False, [1]),
                )

                # NHWC -> NCHW
                out_trans_node = helper.make_node(
                    "Transpose", [dp_node_out], [out_transp_out], perm=[0, 3, 1, 2]
                )
                # insert nodes where the conv is to preserve topological ordering
                graph.node.insert(node_ind, inp_trans_node)
                graph.node.insert(node_ind + 1, dp_conv_node)
                graph.node.insert(node_ind + 2, out_trans_node)
                node_ind += 2
                # remove old nodes
                graph.node.remove(n)
                if has_activation:
                    graph.node.remove(consumer)

                graph_modified = True

        if graph_modified:
            model = model.transform(InferShapes())

        # This transform only requires one pass
        # Also, a second pass would generate unwanted behavior
        return (model, False)
