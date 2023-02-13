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

import pytest

import numpy as np
# as of Feb'20 there is a bug that segfaults ONNX shape inference if we
# import pytorch before onnx, so we make sure to import onnx first
import onnx  # NOQA
from onnx import TensorProto, helper

import finn.core.onnx_exec as oxe
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.infer_doublepacked_dsp import (
    InferDoublePackedConv
)
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from finn.util.basic import gen_finn_dt_tensor


def make_model(ich, och, idim, k, stride, pad, wdt, idt, tdt, odt):

    odim = int((idim+2*pad-k)/stride+1)

    W = np.random.randint(wdt.min(), wdt.max()+1, size=(och, ich, k, k))
    W = W.astype(np.float32)

    if tdt != odt:
        T = np.random.randint(tdt.min(), tdt.max()+1, size=(och, 2**odt.bitwidth()-1))
        T = T.astype(np.float32)
    else:
        T = None

    ishape = [1, ich, idim, idim]
    oshape = [1, och, odim, odim]
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, ishape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, oshape)
    if T is not None:
        inter = helper.make_tensor_value_info("inter", TensorProto.FLOAT, oshape)

    nodes_list = []

    conv_node = helper.make_node(
        "Conv",
        ["inp", "weights"],
        (["outp"] if T is None else ["inter"]),
        kernel_shape=[k, k],
        pads=[pad, pad, pad, pad],
        strides=[stride, stride]
    )
    nodes_list.append(conv_node)

    if T is not None:
        multithresh_node = helper.make_node(
            "MultiThreshold",
            ["inter", "thresh"],
            ["outp"],
            domain="finnexperimental.custom_op.general",
            out_bias=0.0,
            out_scale=1.0,
            out_dtype=odt.name
        )
        nodes_list.append(multithresh_node)

    graph = helper.make_graph(
        nodes=nodes_list, name="conv_graph", inputs=[inp], outputs=[outp]
    )

    model = helper.make_model(graph, producer_name="conv_model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)

    model.set_tensor_datatype("weights", wdt)
    model.set_initializer("weights", W)

    if T is not None:
        model.set_tensor_datatype("inter", tdt)
        model.graph.value_info.append(inter)
        model.set_tensor_datatype("thresh", tdt)
        model.set_initializer("thresh", T)

    return model


@pytest.mark.parametrize("ich", [3])
@pytest.mark.parametrize("och", [64])
@pytest.mark.parametrize("k", [5, 7])
@pytest.mark.parametrize("s", [1, 2])
@pytest.mark.parametrize("pad", [3])
@pytest.mark.parametrize("wdt", [DataType["INT8"], DataType["INT4"]])
@pytest.mark.parametrize("idt", [DataType["UINT8"]])
@pytest.mark.parametrize("tdt", [DataType["INT24"]])
@pytest.mark.parametrize("odt", [DataType["UINT4"]])
@pytest.mark.parametrize("idim", [32, 224])
@pytest.mark.parametrize("mode", ["cppsim", "rtlsim"])
@pytest.mark.slow
@pytest.mark.vivado
def test_fpgadataflow_packed_dsp(ich, och, idim, k, s, pad, wdt, idt, tdt, odt, mode):
    model = make_model(ich, och, idim, k, s, pad, wdt, idt, tdt, odt)
    cdp_model = model.transform(InferDoublePackedConv())
    assert (len(cdp_model.graph.node) == 3 and
            cdp_model.graph.node[1].op_type == "ConvDoublePacked_Batch" and
            cdp_model.graph.node[0].op_type == "Transpose" and
            cdp_model.graph.node[-1].op_type == "Transpose"), "Incorrect model"
    # execute models and compare
    x = gen_finn_dt_tensor(idt, (1, ich, idim, idim))
    input_dict = {"inp": x}
    y_expected = oxe.execute_onnx(model, input_dict)["outp"]

    if mode == "cppsim":
        cdp_model = cdp_model.transform(SetExecMode("cppsim"))
        cdp_model = cdp_model.transform(PrepareCppSim())
        cdp_model = cdp_model.transform(CompileCppSim())
        y_produced = oxe.execute_onnx(cdp_model, input_dict)["outp"]
    elif mode == "rtlsim":
        cdp_model = cdp_model.transform(SetExecMode("rtlsim"))
        cdp_model = cdp_model.transform(GiveUniqueNodeNames())
        cdp_model = cdp_model.transform(GiveReadableTensorNames())
        cdp_model = cdp_model.transform(PrepareIP("xc7z020clg400-1", 5))
        cdp_model = cdp_model.transform(HLSSynthIP())
        cdp_model = cdp_model.transform(PrepareRTLSim())
        input_dict = {"global_in": x}
        y_produced = oxe.execute_onnx(cdp_model, input_dict)["global_out"]

    assert (y_produced.flatten() == y_expected.flatten()).all(), "cppsim failed"
