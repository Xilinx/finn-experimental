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
import sys
import numpy as np
from shutil import copy
import math

from finn.custom_op.fpgadataflow.hlscustomop import HLSCustomOp
from finn.core.datatype import DataType
from onnx import TensorProto, helper
from finn.util.basic import CppBuilder
from finn.util.basic import interleave_matrix_outer_dim_from_partitions
from finn.util.data_packing import (
    npy_to_rtlsim_input,
    numpy_to_hls_code,
    rtlsim_output_to_npy,
)

class ConvDoublePacked_Batch(HLSCustomOp):
    """
    """

    def get_nodeattr_types(self):
        my_attrs = {
            "ConvKernelDim": (
                "i",
                True,
                0,
            ),  # e.g 3 for a 3x3 conv kernel (assumed square)
            "IFMChannels": ("i", True, 0),  # number of input feature maps
            "IFMDim": ("i", True, 0),  # width of input feature map (assumed square)
            "OFMChannels": ("i", True, 0),  # number of output feature maps
            "OFMDim": ("i", True, 0),
            "Stride": ("i", True, 0),
            "Padding": ("i", True, 0),
            # matrix-vector unit parameters
            "SIMD": ("i", True, 0),  # number of SIMD lanes (SIMDWidth)
            "PE": ("i", True, 0),  # number of PEs (PECount)
            "MW": ("i", True, 0),
            "MH": ("i", True, 0),
            # precision parameters
            # FINN DataTypes for inputs, weights, outputs
            "inputDataType": ("s", True, ""),
            "weightDataType": ("s", True, ""),
            # "thresDataType": ("s", False, ""),
            "outputDataType": ("s", True, ""),
            # "MacPrecision": ("i",True,0),      #MAC bitwidth
            "noActivation": ("i", False, 0),  # "ActivationType": ("i",True,0)
            "MMV": ("i", False, 2),  # MMV value, related to output bandwidth (NumVecs)
            # template<int> class type_input  = ap_uint   #For first layer use int value
            # number of input vectors, examples:
            # [1] is a single vector (like a FC layer with batch=1)
            # [4] is four vectors (like a FC layer with batch=4)
            # [1, 4, 4] is four * four vectors (like a conv layer with batch=1)
            "numInputVectors": ("ints", False, [1]),
            # FPGA resource type for ConvolutionInputGenerator input buffer
            # auto -- let Vivado HLS decide
            # block -- use BRAM
            # distributed -- use LUTRAM
            # ultra -- use URAM
            "ram_style": ("s", False, "ultra"),
        }

        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_input_datatype(self):
        """Returns FINN DataType of input."""
        idt = DataType[self.get_nodeattr("inputDataType")]
        assert idt == DataType.UINT8, "inputDataType must be UINT8"
        return idt

    def get_weight_datatype(self):
        """Returns FINN DataType of weights."""
        wdt = DataType[self.get_nodeattr("weightDataType")]
        assert wdt.bitwidth() == 8, "weightDataType 8 bit long"
        return wdt

    # def get_thres_datatype(self):
    #     """Returns FINN DataType of weights."""
    #     return DataType[self.get_nodeattr("thresDataType")]

    def get_output_datatype(self):
        """Returns FINN DataType of output."""
        return DataType[self.get_nodeattr("outputDataType")]

    def get_normal_input_shape(self):
        ifm_dim = self.get_nodeattr("IFMDim")
        ifm_ch = self.get_nodeattr("IFMChannels")

        ishape = (1, ifm_dim, ifm_dim, ifm_ch)
        return ishape

    def get_folded_input_shape(self):
        # ifm_dim = self.get_nodeattr("IFMDim")
        # ifm_ch = self.get_nodeattr("IFMChannels")
        # simd = self.get_nodeattr("SIMD")
        # assert ifm_ch % simd == 0, "SIMD must divide IFMChannels"
        # wf = int(ifm_ch / simd)
        # folded_ishape = (1, ifm_dim, ifm_dim, wf, simd)
        folded_ishape = self.get_normal_input_shape()
        return folded_ishape

    def get_normal_output_shape(self):
        mh = self.get_nodeattr("MH")
        vecs = list(self.get_nodeattr("numInputVectors"))
        normal_output_shape = tuple(vecs + [mh])
        return normal_output_shape

    def get_folded_output_shape(self):
        # mh = self.get_nodeattr("MH")
        # pe = self.get_nodeattr("PE")
        # nf = mh // pe
        # vecs = list(self.get_nodeattr("numInputVectors"))
        # folded_output_shape = tuple(vecs + [nf, pe])
        folded_output_shape = self.get_normal_output_shape()
        return folded_output_shape

    def get_number_output_values(self):
        nf = np.prod(self.get_folded_output_shape()[:-1])
        return nf

    def get_instream_width(self):
        """Returns stream width, input and output stream width are equal for
        the sliding window function"""
        ibits = self.get_input_datatype().bitwidth()
        # simd = self.get_nodeattr("SIMD")
        ifm_ch = self.get_nodeattr("IFMChannels")
        # assert ifm_ch % simd == 0, "SIMD must divide IFMChannels"
        in_width = ifm_ch * ibits
        # in_width = simd * ibits
        return in_width

    def get_outstream_width(self):
        o_bits = self.get_output_datatype().bitwidth()
        # out_width = o_bits * self.get_nodeattr("PE")
        out_width = o_bits * self.get_nodeattr("OFMChannels")
        return out_width

    def make_shape_compatible_op(self, model):
        oshape = self.get_normal_output_shape()
        # implement tensor with correct shape
        values = np.random.randn(*oshape).astype(np.float32)
        return helper.make_node(
            "Constant",
            inputs=[],
            outputs=[self.onnx_node.output[0]],
            value=helper.make_tensor(
                name="const_tensor",
                data_type=TensorProto.FLOAT,
                dims=values.shape,
                vals=values.flatten().astype(float),
            ),
        )

    def infer_node_datatype(self, model):
        node = self.onnx_node
        # data type stays the same
        dtype = self.get_output_datatype()
        model.set_tensor_datatype(node.output[0], dtype)

    def verify_node(self):
        pass

    def bram_estimation(self):
        P = self.get_nodeattr("PE")
        Q = self.get_nodeattr("SIMD")
        wdt = self.get_weight_datatype()
        W = wdt.bitwidth()
        D_in = self.get_nodeattr("MW")
        D_out = self.get_nodeattr("MH")
        omega = (D_in * D_out) / (Q * P)
        mem_width = Q * W * P
        # assuming SDP mode RAMB18s (see UG573 Table 1-10)
        # assuming decoupled (RTL) memory, which is more efficient than const (HLS)
        if mem_width == 1:
            return math.ceil(omega / 16384)
        elif mem_width == 2:
            return math.ceil(omega / 8192)
        elif mem_width <= 4:
            return (math.ceil(omega / 4096)) * (math.ceil(mem_width / 4))
        elif mem_width <= 9:
            return (math.ceil(omega / 2048)) * (math.ceil(mem_width / 9))
        elif mem_width <= 18 or omega > 512:
            return (math.ceil(omega / 1024)) * (math.ceil(mem_width / 18))
        else:
            return (math.ceil(omega / 512)) * (math.ceil(mem_width / 36))

    def bram_efficiency_estimation(self):
        wdt = self.get_weight_datatype()
        W = wdt.bitwidth()
        D_in = self.get_nodeattr("MW")
        D_out = self.get_nodeattr("MH")
        bram16_est = self.bram_estimation()
        if bram16_est == 0:
            return 1
        wbits = W * D_in * D_out
        bram16_est_capacity = bram16_est * 36 * 512
        return wbits / bram16_est_capacity

    def lut_estimation(self):
        P = self.get_nodeattr("PE")
        Q = self.get_nodeattr("SIMD")
        MW = self.get_nodeattr("MW")
        wdt = self.get_weight_datatype()
        W = wdt.bitwidth()
        # determine tdt with input and weight data types
        idt = self.get_input_datatype()
        A = idt.bitwidth()
        # parameters from experiments in paper mentioned above
        c0 = 300
        c1 = 1.1
        # adder tree
        addertree_luts = (W + A) * (2 * Q - 1)
        # accumulator
        acc_bits = W + A + np.ceil(math.log(MW, 2))
        acc_luts = acc_bits
        # thresholds and threshold comparators
        thr_luts = 0
        comp_luts = 0
        noact = self.get_nodeattr("noActivation")
        if noact == 0:
            odt = self.get_output_datatype()
            B = odt.bitwidth()
            thr_luts = (2 ** B - 1) * acc_bits * math.ceil(self.calc_tmem() / 64)
            comp_luts = (2 ** B - 1) * acc_bits

        return int(
            c0 + c1 * (P * (addertree_luts + acc_luts + thr_luts + comp_luts))
        )

    def dsp_estimation(self):
        pe = self.get_nodeattr("PE")
        mmv = self.get_nodeattr("MMV")
        simd = self.get_nodeattr("SIMD")
        return int(pe * mmv * simd / 2)

    def calc_wmem(self):
        """Calculates and returns WMEM."""
        mw = self.get_nodeattr("MW")
        mh = self.get_nodeattr("MH")
        pe = self.get_nodeattr("PE")
        simd = self.get_nodeattr("SIMD")
        assert mh % pe == 0, "Requirement MH divisable by PE is violated."
        assert mw % simd == 0, "Requirement MW divisable by SIMD is violated."
        wmem = mw * mh // (pe * simd)
        return wmem

    def calc_tmem(self):
        """Calculates and returns TMEM."""
        # if self.get_nodeattr("noActivation") == 1:
        #     return 0
        # else:
        mh = self.get_nodeattr("MH")
        pe = self.get_nodeattr("PE")
        return mh // pe

    def read_npy_data(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        dtype = self.get_input_datatype()
        if dtype == DataType.BIPOLAR:
            # use binary for bipolar storage
            dtype = DataType.BINARY
        elem_bits = dtype.bitwidth()
        packed_bits = self.get_instream_width()
        packed_hls_type = "ap_uint<%d>" % packed_bits
        elem_hls_type = dtype.get_hls_datatype_str()
        npy_type = "float"
        npy_in = "%s/input_0.npy" % code_gen_dir
        self.code_gen_dict["$READNPYDATA$"] = []
        self.code_gen_dict["$READNPYDATA$"].append(
            'npy2apintstream<%s, %s, %d, %s>("%s", in0, false);'
            % (packed_hls_type, elem_hls_type, elem_bits, npy_type, npy_in)
        )

    def strm_decl(self):
        self.code_gen_dict["$STREAMDECLARATIONS$"] = []
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> in0 ("in0");'.format(self.get_instream_width())
        )
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> out ("out");'.format(self.get_outstream_width())
        )

    def dataoutstrm(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        dtype = self.get_output_datatype()
        if dtype == DataType.BIPOLAR:
            # use binary for bipolar storage
            dtype = DataType.BINARY
        elem_bits = dtype.bitwidth()
        packed_bits = self.get_outstream_width()
        packed_hls_type = "ap_uint<%d>" % packed_bits
        elem_hls_type = dtype.get_hls_datatype_str()
        npy_type = "float"
        npy_out = "%s/output.npy" % code_gen_dir
        shape = self.get_folded_output_shape()
        shape_cpp_str = str(shape).replace("(", "{").replace(")", "}")

        # note: the innermost dim is not reversed for the output
        self.code_gen_dict["$DATAOUTSTREAM$"] = [
            'apintstream2npy<%s, %s, %d, %s>(out, %s, "%s", false);'
            % (
                packed_hls_type,
                elem_hls_type,
                elem_bits,
                npy_type,
                shape_cpp_str,
                npy_out,
            )
        ]

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "double_packed_conv.hpp"']
        self.code_gen_dict["$GLOBALS$"] += ['#include "params.h"']

        has_thres = self.get_nodeattr("noActivation") == 0
        if has_thres:
            self.code_gen_dict["$GLOBALS$"] += ['#include "thresh.h"']

    def defines(self, var):
        self.code_gen_dict["$DEFINES$"] = []

        # define L0_PE 64
        pe = self.get_nodeattr("PE")
        self.code_gen_dict["$DEFINES$"] += ["#define L0_PE {}".format(pe)]

        # define L0_SIMD 3
        simd = self.get_nodeattr("SIMD")
        self.code_gen_dict["$DEFINES$"] += ["#define L0_SIMD {}".format(simd)]

        # define L0_WMEM 49
        wmen = self.calc_wmem()
        self.code_gen_dict["$DEFINES$"] += ["#define L0_WMEM {}".format(wmen)]

        # define L0_TMEM 1
        tmem = self.calc_tmem()
        self.code_gen_dict["$DEFINES$"] += ["#define L0_TMEM {}".format(tmem)]

        # define L0_MMV 16
        mmv = self.get_nodeattr("MMV")
        self.code_gen_dict["$DEFINES$"] += ["#define L0_MMV {}".format(mmv)]

        # define L0_IFMC 3
        ifmc = self.get_nodeattr("IFMChannels")
        self.code_gen_dict["$DEFINES$"] += ["#define L0_IFMC {}".format(ifmc)]

        # define L0_OFMC 64
        ofmc = self.get_nodeattr("OFMChannels")
        self.code_gen_dict["$DEFINES$"] += ["#define L0_OFMC {}".format(ofmc)]

        # define L0_KERNELDIM 7
        k = self.get_nodeattr("ConvKernelDim")
        self.code_gen_dict["$DEFINES$"] += ["#define L0_KERNELDIM {}".format(k)]

        # define L0_STRIDE 2
        stride = self.get_nodeattr("Stride")
        self.code_gen_dict["$DEFINES$"] += ["#define L0_STRIDE {}".format(stride)]

        # define L0_STRIDE 2
        pad = self.get_nodeattr("Padding")
        self.code_gen_dict["$DEFINES$"] += ["#define Padding {}".format(pad)]

        # define L0_IFMDIM 224
        ifmdim = self.get_nodeattr("IFMDim")
        self.code_gen_dict["$DEFINES$"] += ["#define L0_IFMDIM {}".format(ifmdim)]

        # define L0_OFMDIM 112
        ofmdim = self.get_nodeattr("OFMDim")
        self.code_gen_dict["$DEFINES$"] += ["#define L0_OFMDIM {}".format(ofmdim)]

        # define L0_WBITS 8
        wbits = self.get_weight_datatype().bitwidth()
        self.code_gen_dict["$DEFINES$"] += ["#define L0_WBITS {}".format(wbits)]

        # define L0_ACTBITS 2
        obits = self.get_output_datatype().bitwidth()
        self.code_gen_dict["$DEFINES$"] += ["#define L0_ACTBITS {}".format(obits)]

        # define L0_INBITS 8
        ibits = self.get_input_datatype().bitwidth()
        self.code_gen_dict["$DEFINES$"] += ["#define L0_INBITS {}".format(ibits)]

        # define L0_MACBITS 24
        # macbits = 32
        macbits = 24
        self.code_gen_dict["$DEFINES$"] += ["#define L0_MACBITS {}".format(macbits)]

        # define L0_THBITS 96
        tbits = 32

        has_thres = self.get_nodeattr("noActivation") == 0
        if has_thres:
            num_of_thres = 2 ** self.get_output_datatype().bitwidth() - 1
            self.code_gen_dict["$DEFINES$"] += [
                "#define L0_THBITS {}".format(num_of_thres * tbits)
            ]
            self.code_gen_dict["$DEFINES$"] += [
                "#define ACTIVATION_TYPE FULL_THRESHOLDS"
            ]
        else:
            # L0_THBITS needs to be >0 to avoid compilation errors
            self.code_gen_dict["$DEFINES$"] += ["#define L0_THBITS {}".format(tbits)]
            self.code_gen_dict["$DEFINES$"] += ["#define ACTIVATION_TYPE NO_THRESHOLDS"]
            self.code_gen_dict["$DEFINES$"] += [
                "const ap_uint<L0_THBITS> thres_conv0[L0_PE][L0_TMEM];"
            ]

        numReps = 1  # TODO take it from numInputVectors
        self.code_gen_dict["$DEFINES$"] += ["#define numReps {}".format(numReps)]

    def ipgen_extra_directives(self):
        "Use the extra tcl directives for HLS synthesis to include the extra hpp."
        d = os.path.dirname(sys.modules["finn.custom_op.experimental"].__file__)
        d = os.path.join(d, "../../../../hlslib_extensions")
        return [
            """add_files $config_hwsrcdir/top_%s.cpp -cflags \"-std=c++0x -I%s -I$config_bnnlibdir\""""
             % (self.onnx_node.name, d)
        ]

    def docompute(self):
        # ram_style = self.get_nodeattr("ram_style")
        # map_to_hls_ram_style = {
        #     "auto": "ap_resource_dflt()",
        #     "block": "ap_resource_bram()",
        #     "distributed": "ap_resource_lutram()",
        #     "ultra": "ap_resource_uram()",
        # }
        # hls_ram_style = map_to_hls_ram_style[ram_style]

        self.code_gen_dict["$DOCOMPUTE$"] = [
            """ConvolutionalLayerMMV_Same_Batch_kernel_stride_dsp_packed<L0_KERNELDIM,
            L0_IFMC, L0_IFMDIM, L0_OFMC, L0_STRIDE, Padding, L0_SIMD, L0_PE, L0_WMEM,
            L0_TMEM, L0_WBITS, L0_THBITS, L0_MACBITS, L0_INBITS, L0_ACTBITS, L0_MMV,
             ACTIVATION_TYPE> (in0, out, weights_conv0, thres_conv0, numReps);"""
        ]

    def save_as_npy(self):
        self.code_gen_dict["$SAVEASCNPY$"] = []

    def blackboxfunction(self):
        packed_ibits = self.get_instream_width()
        packed_in_hls_type = "ap_uint<%d>" % packed_ibits

        packed_obits = self.get_outstream_width()
        packed_out_hls_type = "ap_uint<%d>" % packed_obits
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            "void %s(hls::stream<%s > &in0, hls::stream<%s > &out)"
            % (self.onnx_node.name, packed_in_hls_type, packed_out_hls_type)
        ]

    def pragmas(self):
        self.code_gen_dict["$PRAGMAS$"] = [
            "#pragma HLS INTERFACE axis register both port=in0"
        ]
        self.code_gen_dict["$PRAGMAS$"] += [
            "#pragma HLS INTERFACE axis register both port=out"
        ]
        self.code_gen_dict["$PRAGMAS$"] += [
            "#pragma HLS INTERFACE ap_ctrl_none port=return"
        ]
        self.code_gen_dict["$PRAGMAS$"] += ["#pragma HLS DATAFLOW"]

        self.code_gen_dict["$PRAGMAS$"] += [
            "#pragma HLS ARRAY_PARTITION variable=weights_conv0 complete dim=1"
        ]
        self.code_gen_dict["$PRAGMAS$"] += [
            "#pragma HLS RESOURCE variable=weights_conv0 core=ROM_1P_LUTRAM"
        ]
        self.code_gen_dict["$PRAGMAS$"] += [
            "#pragma HLS ARRAY_PARTITION variable=thres_conv0 complete dim=1"
        ]
        self.code_gen_dict["$PRAGMAS$"] += [
            "#pragma HLS RESOURCE variable=thres_conv0 core=ROM_1P_LUTRAM"
        ]

    def get_hls_compatible_weight_tensor(self, orig_weight_matrix):
        """Convert the original numpy weight matrix orig_weight_matrix into
        a form suitable for passing to the hlslib call:
        * ensure MH % PE == 0 and MW % SIMD == 0
        * for bipolar {-1,+1} weights, convert to binary {0, 1}
        * interleave rows between PEs
        * reshape into (1, PE, WMEM, SIMD) and return
        """
        mw = self.get_nodeattr("MW")
        mh = self.get_nodeattr("MH")
        pe = self.get_nodeattr("PE")
        simd = self.get_nodeattr("SIMD")
        wmem = self.calc_wmem()
        assert orig_weight_matrix.shape == (
            mw,
            mh,
        ), """Weights matrix doesn't
        have expected shape (mw, mh)"""
        assert mw % simd == 0, "Requirement MH divisable by SIMD is violated."
        assert mh % pe == 0, "Requirement MH divisable by PE is violated."
        # start by transposing the original weight matrix, since ONNX and
        # finn-hlslib use different assumptions
        # ONNX uses (in_features, out_features) and matmul(x, W)
        # finn-hlslib uses (out_features, in_features) and matmul(W, x)
        ret = orig_weight_matrix.T
        if self.get_weight_datatype() == DataType.BIPOLAR:
            # convert bipolar to binary
            ret = (ret + 1) / 2
        # interleave rows between PEs and reshape
        # distribute rows between PEs
        ret = interleave_matrix_outer_dim_from_partitions(ret, pe)
        # create SIMD as innermost dimension and add a dummy outer dim
        ret = ret.reshape(1, pe, wmem, simd)
        # reverse the SIMD dimension
        ret = np.flip(ret, axis=-1)
        return ret

    def get_hls_compatible_threshold_tensor(self, orig_thres_matrix):
        """Convert the original numpy weight matrix orig_weight_matrix into
        a form suitable for passing to the hlslib call:
        * ensure MH % PE == 0
        * for bipolar weights&inputs, ensure thresholds are positive
        * interleave rows between PEs
        * reshape into (PE, TMEM, n_thres_steps) and return
        """
        mh = self.get_nodeattr("MH")
        pe = self.get_nodeattr("PE")
        tmem = mh // pe
        assert mh % pe == 0, "Requirement MH divisable by PE is violated."
        assert (
            orig_thres_matrix.ndim == 2
        ), """Threshold matrix dimension is
        not as expected (2)."""
        n_thres_steps = orig_thres_matrix.shape[1]
        # inp_is_bipolar = self.get_input_datatype() == DataType.BIPOLAR
        # wt_is_bipolar = self.get_weight_datatype() == DataType.BIPOLAR
        # # reinterpret inp/wt as bipolar if bin_xnor_mode is iset
        # inp_is_binary = self.get_input_datatype() == DataType.BINARY
        # wt_is_binary = self.get_weight_datatype() == DataType.BINARY
        # bin_xnor_mode = self.get_nodeattr("binaryXnorMode") == 1
        # inp_is_bipolar = inp_is_bipolar or (inp_is_binary and bin_xnor_mode)
        # wt_is_bipolar = wt_is_bipolar or (wt_is_binary and bin_xnor_mode)
        # if inp_is_bipolar and wt_is_bipolar:
        #     # ensure all thresholds are nonnegative
        #     assert (orig_thres_matrix >= 0).all()
        # ensure all thresholds are integer
        assert (orig_thres_matrix.astype(np.int32) == orig_thres_matrix).all()
        ret = orig_thres_matrix
        # ensure channels = mh , duplicating if necessary
        if ret.shape[0] == 1:
            ret = np.tile(ret, (mh, 1))
        assert (
            ret.shape[0] == mh
        ), "Channels of threshold matrix are not as expected (mh)"
        # distribute rows between PEs
        ret = interleave_matrix_outer_dim_from_partitions(ret, pe)
        assert (
            ret.shape[0] == pe
        ), """First dimension after distribution of the
        rows between PEs is not as expected (pe)"""
        assert (
            ret.shape[1] == tmem
        ), """Second dimension after distribution of the
        rows between PEs is not as expected (tmem)"""
        assert (
            ret.shape[2] == n_thres_steps
        ), """Third dimension after distribution of the
        rows between PEs is not as expected (n_thres_steps)"""
        return ret.reshape(1, pe, tmem, n_thres_steps)

    def get_ap_int_max_w(self):
        instream = self.get_instream_width()
        outstream = self.get_outstream_width()
        omvau = outstream * self.get_nodeattr("MMV")
        return max([instream, omvau])

    def generate_params(self, model, path):

        # weights
        weights = model.get_initializer(self.onnx_node.input[1])
        # convert weights into hlslib-compatible format
        weight_tensor = self.get_hls_compatible_weight_tensor(weights)
        wdt = self.get_weight_datatype()
        # we have converted bipolar weights to binary for export,
        # so use it as such for weight generation

        # if self.get_weight_datatype() == DataType.BIPOLAR:
        #     wdt = DataType.BINARY

        code_gen_dir = path

        """Saves weights into params.h"""
        weight_hls_code = numpy_to_hls_code(weight_tensor, wdt, "weights", True, True)
        # remove extra {}
        weight_hls_code = weight_hls_code[1:-2] + weight_hls_code[-1:]

        weight_hls_code.replace("{{{", "{{").replace("}}}", "}}")
        # write weights into params.h
        f_weights = open("{}/params.h".format(code_gen_dir), "w")

        wbits = wdt.bitwidth()

        # if wdt.bitwidth() != 1:
        f_weights.write(
            "const ap_uint<{}> weights_conv0[{}][{}] =  ".format(
                self.get_nodeattr("SIMD") * wbits,
                self.get_nodeattr("PE"),
                self.calc_wmem(),
            )
        )

        f_weights.write(weight_hls_code)
        f_weights.close()

        # save thresholds in thresh.h
        has_thres = self.get_nodeattr("noActivation") == 0
        if has_thres:
            assert len(self.onnx_node.input) > 2, "need at least 3 inputs"
            thresholds = model.get_initializer(self.onnx_node.input[2])
            if thresholds is not None:
                threshold_tensor = self.get_hls_compatible_threshold_tensor(thresholds)
                tdt = DataType.INT32

                thresholds_hls_code = numpy_to_hls_code(
                    threshold_tensor, tdt, "thresholds", True, True
                )
                thresholds_hls_code = (
                    thresholds_hls_code[1:-2] + thresholds_hls_code[-1:]
                )
                # thresholds_hls_code.replace("{{{", "{{").replace("}}}", "}}")
                # write thresholds into thresh.h
                f_thresh = open("{}/thresh.h".format(code_gen_dir), "w")
                # tdt_hls = tdt.get_hls_datatype_str()
                # use binary to export bipolar activations
                # export_odt = self.get_output_datatype()
                # if self.get_output_datatype() == DataType.BIPOLAR:
                #     export_odt = DataType.BINARY
                # odt_hls = export_odt.get_hls_datatype_str()

                f_thresh.write(
                    "const ap_uint<{}> thres_conv0[{}][{}] = ".format(
                        tdt.bitwidth() * threshold_tensor.shape[-1],
                        self.get_nodeattr("PE"),
                        self.calc_tmem(),
                    )
                )
                f_thresh.write(thresholds_hls_code)
                f_thresh.close()
            else:
                print("Warning Attribute noActivation==0 but no threshold initializer")

    def compile_singlenode_code(self):
        """Builds the bash script for compilation using the CppBuilder from
        finn.util.basic and executes the script to produce the executable."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        builder = CppBuilder()
        # to enable additional debug features please uncommand the next line
        # builder.append_includes("-DDEBUG")
        builder.append_includes("-I/workspace/finn/src/finn/qnn-data/cpp")
        builder.append_includes("-I/workspace/cnpy/")
        builder.append_includes("-I/workspace/finn-hlslib")
        builder.append_includes("-I{}/include".format(os.environ["VIVADO_PATH"]))
        # include also the cpp definition for doublepacked conv
        d = os.path.dirname(sys.modules["finn.custom_op.experimental"].__file__)
        d = os.path.join(d, "../../../../hlslib_extensions")
        builder.append_includes("-I%s" % d)
        builder.append_includes("--std=c++11")
        builder.append_includes("-O3")
        builder.append_sources(code_gen_dir + "/*.cpp")
        builder.append_sources("/workspace/cnpy/cnpy.cpp")
        builder.append_includes("-lz")
        builder.set_executable_path(code_gen_dir + "/node_model")
        builder.build(code_gen_dir)
        self.set_nodeattr("executable_path", builder.executable_path)

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        node = self.onnx_node
        exp_ishape = self.get_normal_input_shape()
        exp_oshape = self.get_normal_output_shape()
        folded_ishape = self.get_folded_input_shape()
        folded_oshape = self.get_folded_output_shape()

        # TODO ensure codegen dir exists
        if mode == "cppsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        elif mode == "rtlsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        else:
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following value ("cppsim", "rtlsim")""".format(
                    mode
                )
            )

        inp = context[node.input[0]]
        assert str(inp.dtype) == "float32", "Input datatype is not float32"
        assert (
            inp.shape == exp_ishape
        ), """Input shape doesn't
        match expected shape (1, ifm_dim, ifm_dim, ifm_ch)."""
        if self.get_input_datatype() == DataType.BIPOLAR:
            # store bipolar activations as binary
            inp = (inp + 1) / 2
            export_idt = DataType.BINARY
        else:
            export_idt = self.get_input_datatype()
        # reshape input into folded form
        inp = inp.reshape(folded_ishape)
        # make copy before saving array
        reshaped_input = inp.copy()
        np.save(os.path.join(code_gen_dir, "input_0.npy"), reshaped_input)

        if mode == "cppsim":
            # execute the precompiled model
            super().exec_precompiled_singlenode_model()
            # load output npy file
            super().npy_to_dynamic_output(context)
            assert (
                context[node.output[0]].shape == folded_oshape
            ), "cppsim did not produce expected folded output shape"
            context[node.output[0]] = context[node.output[0]].reshape(*exp_oshape)
        elif mode == "rtlsim":
            sim = self.get_rtlsim()
            nbits = self.get_instream_width()
            rtlsim_inp = npy_to_rtlsim_input(
                "{}/input_0.npy".format(code_gen_dir), export_idt, nbits
            )
            super().reset_rtlsim(sim)
            super().toggle_clk(sim)
            rtlsim_output = self.rtlsim(sim, rtlsim_inp)
            odt = self.get_output_datatype()
            target_bits = odt.bitwidth()
            packed_bits = self.get_outstream_width()
            out_npy_path = "{}/output.npy".format(code_gen_dir)
            out_shape = self.get_folded_output_shape()
            rtlsim_output_to_npy(
                rtlsim_output, out_npy_path, odt, out_shape, packed_bits, target_bits
            )
            # load and reshape output
            output = np.load(out_npy_path)
            output = np.asarray([output], dtype=np.float32).reshape(*exp_oshape)
            context[node.output[0]] = output
        else:
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following value ("cppsim", "rtlsim")""".format(
                    mode
                )
            )

        assert (
            context[node.output[0]].shape == exp_oshape
        ), """Output
        shape doesn't match expected shape (1, ofm_dim, ofm_dim, k*k*ifm_ch)."""
