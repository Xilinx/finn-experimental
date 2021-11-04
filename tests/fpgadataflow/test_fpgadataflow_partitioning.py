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

import random
import math
import pytest
import numpy as np
from onnx import TensorProto, helper

from finn.core.datatype import DataType
from finn.custom_op.registry import getCustomOp
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.analysis.partitioning import partition
from finn.transformation.general import ApplyConfig

def make_multi_fclayer_model(ch, wdt, adt, tdt, nnodes):

    W = np.random.randint(wdt.min(), wdt.max()+1, size=(ch, ch))
    W = W.astype(np.float32)

    T = np.random.randint(tdt.min(), tdt.max()+1, size=(ch, 2**adt.bitwidth()-1))
    T = T.astype(np.float32)

    tensors = []
    tensors.append(helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, ch]))
    for i in range(1, nnodes):
        inter = helper.make_tensor_value_info("inter_"+str(i), TensorProto.FLOAT, [1, ch])
        tensors.append(inter)
    tensors.append(helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, ch]))
        
    FCLayer_nodes = []
    for i in range(nnodes):
        pe = int(random.choice([ch/2, ch/4]))
        simd = int(random.choice([ch/2, ch/4]))
        assert ch % pe == 0
        assert ch % simd == 0
        FCLayer_nodes += [helper.make_node(
            "StreamingFCLayer_Batch",
            [tensors[i].name, "weights_"+str(i), "thresh_"+str(i)],
            [tensors[i+1].name],
            domain="finn.custom_op.fpgadataflow",
            backend="fpgadataflow",
            MW=ch,
            MH=ch,
            SIMD=simd,
            PE=pe,
            inputDataType=adt.name,
            weightDataType=wdt.name,
            outputDataType=adt.name,
            ActVal=0,
            binaryXnorMode=0,
            noActivation=0,
            inFIFODepth=int(2 ** math.ceil(math.log2(random.randint(1,1000)))),
            outFIFODepth=int(2 ** math.ceil(math.log2(random.randint(1,1000)))),
        )]

    graph = helper.make_graph(
        nodes=FCLayer_nodes, name="fclayer_graph", inputs=[tensors[0]], outputs=[tensors[-1]]
    )

    model = helper.make_model(graph, producer_name="fclayer-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", adt)
    model.set_tensor_datatype("outp", adt)
    
    for i in range(1, nnodes+1):
        model.graph.value_info.append(tensors[i])
        model.set_initializer("weights_"+str(i-1), W)
        model.set_initializer("thresh_"+str(i-1), T)
        model.set_tensor_datatype("weights_"+str(i-1), wdt)
        model.set_tensor_datatype("thresh_"+str(i-1), tdt)

    return model

@pytest.mark.parametrize("ch", [64])
@pytest.mark.parametrize("wdt", [DataType["INT2"]])
@pytest.mark.parametrize("adt", [DataType["UINT4"]])
@pytest.mark.parametrize("tdt", [DataType["INT16"]])
@pytest.mark.parametrize("nnodes", [5, 20, 200])
@pytest.mark.parametrize("platform", ["U50", "U250"])
def test_partitioning_singledevice(ch, wdt, adt, tdt, nnodes, platform):
    model = make_multi_fclayer_model(ch, wdt, adt, tdt, nnodes)
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InsertDWC())
    model = model.transform(InsertFIFO())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    #apply partitioning
    floorplan = partition(model, 5, platform)
    if floorplan is not None:
        assert len(floorplan) == 1
        floorplan = floorplan[0]
    if nnodes == 200:
        assert floorplan is None
        return
    model = model.transform(ApplyConfig(floorplan))
    # check the SLR assignments of each block and 
    # count the number of SLRs required
    counts = dict()
    for node in model.graph.node:
        nodeInst = getCustomOp(node)
        assert nodeInst.get_nodeattr("slr") != -1
        slr = nodeInst.get_nodeattr("slr")
        counts["SLR"+str(slr)] = counts.get("SLR"+str(slr),0) + 1
    # check against expectations
    if nnodes < 15:
        assert len(counts.keys()) == 1
    else:
        assert len(counts.keys()) > 1

@pytest.mark.parametrize("ch", [64])
@pytest.mark.parametrize("wdt", [DataType["INT2"]])
@pytest.mark.parametrize("adt", [DataType["UINT4"]])
@pytest.mark.parametrize("tdt", [DataType["INT16"]])
@pytest.mark.parametrize("nnodes", [40])
@pytest.mark.parametrize("platform", ["U50"])
def test_partitioning_multidevice(ch, wdt, adt, tdt, nnodes, platform):
    model = make_multi_fclayer_model(ch, wdt, adt, tdt, nnodes)
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InsertDWC())
    model = model.transform(InsertFIFO())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    #apply partitioning
    floorplan = partition(model, 5, platform, ndevices=2)
    if floorplan is not None:
        assert len(floorplan) == 1
    floorplan = floorplan[0]
    model = model.transform(ApplyConfig(floorplan))
    # check the device assignments of each block and 
    # count the number of devices required
    counts = dict()
    for node in model.graph.node:
        nodeInst = getCustomOp(node)
        slr = nodeInst.get_nodeattr("device_id")
        counts["dev"+str(slr)] = counts.get("dev"+str(slr),0) + 1
    # check against expectations
    assert len(counts.keys()) > 1
    
@pytest.mark.parametrize("ch", [64])
@pytest.mark.parametrize("wdt", [DataType["INT2"]])
@pytest.mark.parametrize("adt", [DataType["UINT4"]])
@pytest.mark.parametrize("tdt", [DataType["INT16"]])
@pytest.mark.parametrize("nnodes", [10])
@pytest.mark.parametrize("platform", ["U50"])
def test_partitioning_multireplica(ch, wdt, adt, tdt, nnodes, platform):
    model = make_multi_fclayer_model(ch, wdt, adt, tdt, nnodes)
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InsertDWC())
    model = model.transform(InsertFIFO())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    #apply partitioning
    floorplan = partition(model, 5, platform, nreplicas=2)
    assert len(floorplan) == 2
