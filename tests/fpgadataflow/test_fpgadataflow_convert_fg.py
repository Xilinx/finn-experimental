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

from onnx import TensorProto, helper
import numpy as np
import pytest

from finn.core.datatype import DataType
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.general import GiveUniqueNodeNames
from finn.transformation.lower_convs_to_matmul import LowerConvsToMatMul

from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
import finn.core.onnx_exec as oxe
from finn.core.modelwrapper import ModelWrapper
from finn.util.basic import gen_finn_dt_tensor
import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls

from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.custom_op.general.im2col import compute_conv_output_dim
from finn.custom_op.registry import getCustomOp
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer

from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.make_finegrained import MakeFinegrained

from finn.util.test import (
    get_trained_network_and_ishape,
    load_test_checkpoint_or_skip,
)
import brevitas.onnx as bo
import os

from finn.transformation.general import (
    RemoveUnusedTensors,
    RemoveStaticGraphInputs,
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
)
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.streamline import Streamline
from finn.transformation.fpgadataflow.annotate_resources import AnnotateResources
from finn.transformation.infer_data_layouts import InferDataLayouts
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
from finn.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from finn.transformation.streamline.reorder import (
    MakeMaxPoolNHWC,
    MoveScalarLinearPastInvariants,
)
import finn.transformation.streamline.absorb as absorb
from finn.transformation.fold_constants import FoldConstants
from finn.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
from finn.transformation.fpgadataflow.set_folding import SetFolding
from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO

def fold_lfc(model):
    fc_layers = model.get_nodes_by_op_type("StreamingFCLayer_Batch")
    # (PE, SIMD, ramstyle) for each layer
    config = [
        (32, 49, "block"),
        (64, 32, "auto"),
        (32, 64, "auto"),
        (10, 8, "distributed"),
    ]
    for fcl, (pe, simd, ramstyle) in zip(fc_layers, config):
        fcl_inst = getCustomOp(fcl)
        fcl_inst.set_nodeattr("PE", pe)
        fcl_inst.set_nodeattr("SIMD", simd)
        fcl_inst.set_nodeattr("ram_style", ramstyle)
        fcl_inst.set_nodeattr("runtime_writeable_weights", 1)

    thr_layers = model.get_nodes_by_op_type("Thresholding_Batch")
    for i in range(len(thr_layers)):
        thr_inst = getCustomOp(thr_layers[i])
        pe = config[i][0]
        thr_inst.set_nodeattr("PE", pe)
    return model


def fold_cnv(model):
    fc_layers = model.get_nodes_by_op_type("StreamingFCLayer_Batch")
    # each tuple is (PE, SIMD) for a layer
    folding = [
        (16, 3),
        (32, 32),
        (16, 32),
        (16, 32),
        (4, 32),
        (1, 32),
        (1, 4),
        (1, 8),
        (5, 1),
    ]
    for fcl, (pe, simd) in zip(fc_layers, folding):
        fcl_inst = getCustomOp(fcl)
        fcl_inst.set_nodeattr("PE", pe)
        fcl_inst.set_nodeattr("SIMD", simd)

    swg_layers = model.get_nodes_by_op_type("ConvolutionInputGenerator")
    for i in range(len(swg_layers)):
        swg_inst = getCustomOp(swg_layers[i])
        simd = folding[i][1]
        swg_inst.set_nodeattr("SIMD", simd)

    thr_layers = model.get_nodes_by_op_type("Thresholding_Batch")
    for i in range(len(thr_layers)):
        thr_inst = getCustomOp(thr_layers[i])
        pe = folding[i][0]
        thr_inst.set_nodeattr("PE", pe)
    return model


@pytest.mark.parametrize(
    "bnn_config", [("cnv", 1, 1), ("lfc", 1, 1)]#, ("mobilenet", 4, 4)]
)
@pytest.mark.parametrize(
    "mmode", ["const", "decoupled", "external"]
)
@pytest.mark.parametrize(
    "ext_act", [True, False]
)
def test_convert_to_finegrained(bnn_config, mmode, ext_act):
    net, wb, ab = bnn_config

    (model, ishape) = get_trained_network_and_ishape(net, wb, ab)
    chkpt_name = os.environ["FINN_BUILD_DIR"] + "/end2end_%s_w%da%d.onnx" % (net, wb, ab)
    bo.export_finn_onnx(model, ishape, chkpt_name)
    model = load_test_checkpoint_or_skip(chkpt_name)

    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(RemoveStaticGraphInputs())
    model = model.transform(absorb.AbsorbSignBiasIntoMultiThreshold())
    model = model.transform(MoveScalarLinearPastInvariants())
    model = model.transform(Streamline())
    model = model.transform(LowerConvsToMatMul())
    model = model.transform(MakeMaxPoolNHWC())
    model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
    model = model.transform(ConvertBipolarMatMulToXnorPopcount())
    model = model.transform(Streamline())

    model = model.transform(to_hls.InferConvInpGen())
    if ext_act:
        model = model.transform(to_hls.InferThresholdingLayer())

    model = model.transform(to_hls.InferVVAU())
    model = model.transform(to_hls.InferStreamingMaxPool())
    model = model.transform(to_hls.InferQuantizedStreamingFCLayer(mmode))
    model = model.transform(to_hls.InferBinaryStreamingFCLayer(mmode))

    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    model = model.transform(RemoveCNVtoFCFlatten())
    model = model.transform(CreateDataflowPartition())
    for node in model.graph.node:
        if node.op_type == "StreamingDataflowPartition":
            model = ModelWrapper(getCustomOp(node).get_nodeattr("model"))

    # fold
    if net == "cnv":
        model = fold_cnv(model)
    elif net == "lfc":
        model = fold_lfc(model)

    # force FIFOs
    for node in model.graph.node:
        getCustomOp(node).set_nodeattr("outFIFODepth",128)
        getCustomOp(node).set_nodeattr("inFIFODepth",128)

    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InsertDWC())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InsertFIFO())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())

    import pdb; pdb.set_trace()

    model = model.transform(MakeFinegrained())


    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(PrepareIP("xc7z020clg400-1", 5))
    model = model.transform(HLSSynthIP())
    model = model.transform(CreateStitchedIP("xc7z020clg400-1", 5))
