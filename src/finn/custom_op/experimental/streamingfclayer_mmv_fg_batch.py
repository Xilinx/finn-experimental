
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

import warnings
import math
import os, sys
import numpy as np
import subprocess
from onnx import TensorProto, helper
from finn.core.datatype import DataType
from finn.custom_op.fpgadataflow.hlscustomop import HLSCustomOp
from finn.util.basic import (
    make_build_dir,
    interleave_matrix_outer_dim_from_partitions,
    roundup_to_integer_multiple,
    calculate_matvec_accumulator_range,
)
from finn.util.data_packing import (
    npy_to_rtlsim_input,
    numpy_to_hls_code,
    rtlsim_output_to_npy,
    pack_innermost_dim_as_hex_string,
)
from finn.util.ipi_axis_stitch import (
    axis_gather_bcast_scatter,
)
import textwrap

class StreamingFCLayer_MMV_FG_Batch(HLSCustomOp):

    def __init__(self, onnx_node):
        super().__init__(onnx_node)

    def get_nodeattr_types(self):
        my_attrs = {
            "PE": ("i", True, 0),
            "SIMD": ("i", True, 0),
            "MW": ("i", True, 0),
            "MH": ("i", True, 0),
            "resType": ("s", False, "lut", {"auto", "lut", "dsp"}),
            "ActVal": ("i", False, 0),
            # FINN DataTypes for inputs, weights, outputs
            "inputDataType": ("s", True, ""),
            "weightDataType": ("s", True, ""),
            "outputDataType": ("s", True, ""),
            # FINN DataType for accumulator -- auto-computed and updated
            "accDataType": ("s", False, "INT32"),
            # use xnor-popcount for binary weights/inputs, thus treating them
            # as bipolar
            "binaryXnorMode": ("i", False, 0, {0, 1}),
            # number of input vectors, examples:
            # [1] is a single vector (like a FC layer with batch=1)
            # [4] is four vectors (like a FC layer with batch=4)
            # [1, 4, 4] is four * four vectors (like a conv layer with batch=1)
            "numInputVectors": ("ints", False, [1]),
            # memory mode for the FC weights
            # decoupled -- streaming weights with weight streamer packaged inside IP
            # external -- streaming weights with external streamer
            "mem_mode": ("s", False, "decoupled", {"decoupled", "external"}),
            # FPGA resource type for memories in decoupled mode
            # auto -- let Vivado decide
            # block -- use BRAM
            # distributed -- use LUTRAM
            # ultra -- use UltraRAM (URAM), must have runtime_writeable_weights=1
            # see also https://www.xilinx.com/support/answers/38070.html
            "ram_style": (
                "s",
                False,
                "auto",
                {"auto", "block", "distributed", "ultra"},
            ),
            "ibuf_ram_style": (
                "s",
                False,
                "auto",
                {"auto", "block", "distributed", "ultra"},
            ),
            "MMV" : ("i", False, 1),
            "VVAU": ("i", False, 0, {0, 1}),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

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

    def uram_estimation(self):
        P = self.get_nodeattr("PE")
        Q = self.get_nodeattr("SIMD")
        wdt = self.get_weight_datatype()
        W = wdt.bitwidth()
        D_in = self.get_nodeattr("MW")
        D_out = self.get_nodeattr("MH")
        omega = (D_in * D_out) / (Q * P)
        mem_width = Q * W * P
        mmode = self.get_nodeattr("mem_mode")
        mstyle = self.get_nodeattr("ram_style")
        if (mmode == "decoupled" and mstyle != "ultra"):
            return 0
        width_multiplier = math.ceil(mem_width / 72)
        depth_multiplier = math.ceil(omega / 4096)
        return width_multiplier * depth_multiplier

    def bram_estimation(self):
        """Calculates resource estimation for BRAM based on:
        - FINN-R: An End-to-End Deep-Learning Framework for Fast
        Exploration of Quantized Neural Networks
        - M. Blott, T. B. Preusser, N. J. Fraser, G. Gambardella, K. O'Brien,
        Y. Umuroglu, M. Leeser and K. Vissers
        - 12. Sep 2018
        """
        # TODO add in/out FIFO contributions
        P = self.get_nodeattr("PE")
        Q = self.get_nodeattr("SIMD")
        wdt = self.get_weight_datatype()
        W = wdt.bitwidth()
        D_in = self.get_nodeattr("MW")
        D_out = self.get_nodeattr("MH")
        omega = (D_in * D_out) / (Q * P)
        mem_width = Q * W * P
        mmode = self.get_nodeattr("mem_mode")
        mstyle = self.get_nodeattr("ram_style")
        if (mmode == "decoupled" and mstyle in ["distributed", "ultra"]):
            return 0
        # assuming SDP mode RAMB18s (see UG573 Table 1-10)
        # assuming decoupled (RTL) memory
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

    def uram_efficiency_estimation(self):
        """Function for URAM efficiency estimation: actual parameter storage
        needed divided by the allocated URAM storage (from estimation)"""
        wdt = self.get_weight_datatype()
        W = wdt.bitwidth()
        D_in = self.get_nodeattr("MW")
        D_out = self.get_nodeattr("MH")
        uram_est = self.uram_estimation()
        if uram_est == 0:
            return 1
        wbits = W * D_in * D_out
        uram_est_capacity = uram_est * 72 * 4096
        return wbits / uram_est_capacity

    def lut_estimation(self):
        """Calculates resource estimations for LUTs based on:
        - FINN-R: An End-to-End Deep-Learning Framework for Fast
        Exploration of Quantized Neural Networks
        - M. Blott, T. B. Preusser, N. J. Fraser, G. Gambardella, K. O'Brien,
        Y. Umuroglu, M. Leeser and K. Vissers
        - 12. Sep 2018
        """
        # TODO add in/out FIFO contributions
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
        c2 = 0
        mmode = self.get_nodeattr("mem_mode")
        mstyle = self.get_nodeattr("ram_style")
        if (mmode == "decoupled" and mstyle == "distributed"):
            c2 = (P * Q * W) * math.ceil(self.calc_wmem() / 64)

        # multiplication
        res_type = self.get_nodeattr("resType")
        if res_type == "dsp":
            mult_luts = 0
        else:
            mult_luts = Q * (2 * math.ceil((W + A) / 6) - 1) * (W + A)
        # adder tree
        addertree_luts = (W + A) * (2 * Q - 1)
        # accumulator
        acc_bits = W + A + np.ceil(math.log(MW, 2))
        acc_luts = acc_bits
        # thresholds and threshold comparators
        thr_luts = 0
        comp_luts = 0

        return int(
            c0
            + c1 * (P * (mult_luts + addertree_luts + acc_luts + thr_luts + comp_luts))
            + c2
        )

    def dsp_estimation(self):
        # multiplication
        P = self.get_nodeattr("PE")
        res_type = self.get_nodeattr("resType")
        Q = self.get_nodeattr("SIMD")
        wdt = self.get_weight_datatype()
        W = wdt.bitwidth()
        idt = self.get_input_datatype()
        A = idt.bitwidth()
        if res_type == "dsp":
            mult_dsp = P * Q * np.ceil((W + A) / 48)  # TODO: more accurate modelling
        else:
            mult_dsp = 0
        return int(mult_dsp)

    def get_exp_cycles(self):
        mmv = self.get_nodeattr("MMV")
        return super().get_exp_cycles() // mmv

    def get_instream_width(self):
        i_bits = self.get_input_datatype().bitwidth()
        in_width = i_bits * self.get_nodeattr("SIMD") 
        return in_width
    #mod-fine
    def get_instream_width_padded(self):
        i_bits = self.get_input_datatype().bitwidth()
        in_width = i_bits * self.get_nodeattr("SIMD") 
        return roundup_to_integer_multiple(in_width, 8)

    def get_outstream_width(self):

        o_bits = self.get_output_datatype().bitwidth()
        pe = self.get_nodeattr("PE")

        out_width = o_bits * pe
        return out_width

    def get_weightstream_width(self):
        """Returns weight stream width. Used only in decoupled mode."""           
        pe = self.get_nodeattr("PE")
        simd = self.get_nodeattr("SIMD")
        wp = self.get_weight_datatype().bitwidth()
        w_width = pe * simd * wp
        return w_width

    def get_weightstream_width_padded(self):
        """Returns weight stream width padded to a multiple of 8. This is required
        by the AXI Stream spec. Used in decoupled mode."""
        weight_width = self.get_weightstream_width()
        return roundup_to_integer_multiple(weight_width, 8)

    # mod-fine
    def get_weight_splitter_output_width_padded(self):
        weight_width = self.get_weightstream_width()
        pe = self.get_nodeattr("PE")
        weight_width_per_pe = (weight_width/pe)
        return roundup_to_integer_multiple(weight_width_per_pe, 8)

    def get_ap_int_max_w(self):
        # base class impl (max of inp/out stream widths)
        max_of_io = super().get_ap_int_max_w()
        # decoupled mode weight stream
        weightstream = self.get_weightstream_width()//self.get_nodeattr("PE")

        # single PE weight entry
        weight_bits = self.get_weight_datatype().bitwidth()
        simd = self.get_nodeattr("SIMD")
        single_pe_w = simd * weight_bits
        return max([weightstream, max_of_io, single_pe_w])

    def get_folded_input_shape(self):
        mw = self.get_nodeattr("MW")
        simd = self.get_nodeattr("SIMD")
        sf = mw // simd
        vecs = list(self.get_nodeattr("numInputVectors"))
        folded_input_shape = tuple(vecs + [sf, simd])
        return folded_input_shape

    def get_folded_output_shape(self):
        mh = self.get_nodeattr("MH")
        pe = self.get_nodeattr("PE")
        nf = mh // pe
        vecs = list(self.get_nodeattr("numInputVectors"))
        folded_output_shape = tuple(vecs + [nf, pe])
        return folded_output_shape

    def get_normal_input_shape(self):
        mw = self.get_nodeattr("MW")
        vecs = list(self.get_nodeattr("numInputVectors"))
        normal_input_shape = tuple(vecs + [mw])
        return normal_input_shape

    def get_normal_output_shape(self):
        mh = self.get_nodeattr("MH")
        vecs = list(self.get_nodeattr("numInputVectors"))
        normal_output_shape = tuple(vecs + [mh])
        return normal_output_shape

    def get_number_output_values(self):
        nf = np.prod(self.get_folded_output_shape()[:-1])
        return nf

    def get_template_param_values(self):
        """Returns the template parameter values according to input, output and weight
        data types."""
        ret = dict()
        inp_hls_str = self.get_input_datatype().get_hls_datatype_str()
        out_hls_str = self.get_output_datatype().get_hls_datatype_str()
        inp_is_binary = self.get_input_datatype() == DataType.BINARY
        # out_is_binary = self.get_output_datatype() == DataType.BINARY
        wt_is_binary = self.get_weight_datatype() == DataType.BINARY
        bin_xnor_mode = self.get_nodeattr("binaryXnorMode") == 1
        if (inp_is_binary or wt_is_binary) and (not bin_xnor_mode):
            raise Exception("True binary (non-bipolar) inputs not yet supported")
        inp_is_bipolar = self.get_input_datatype() == DataType.BIPOLAR
        # out_is_bipolar = self.get_output_datatype() == DataType.BIPOLAR
        wt_is_bipolar = self.get_weight_datatype() == DataType.BIPOLAR
        # reinterpret inp/wt as bipolar if bin_xnor_mode is iset
        inp_is_bipolar = inp_is_bipolar or (inp_is_binary and bin_xnor_mode)
        wt_is_bipolar = wt_is_bipolar or (wt_is_binary and bin_xnor_mode)
        # fill in TSrcI and TWeightI
        # TODO check these with Giulio
        # TODO handle non-bipolar binary inputs
        if inp_is_bipolar and wt_is_bipolar:
            ret["TSrcI"] = "Recast<XnorMul>"
            ret["TWeightI"] = "Identity"
        elif (not inp_is_bipolar) and wt_is_bipolar:
            ret["TSrcI"] = "Slice<%s>" % inp_hls_str
            ret["TWeightI"] = "Recast<Binary>"
        elif inp_is_bipolar and (not wt_is_bipolar):
            ret["TSrcI"] = "Recast<Binary>"
            ret["TWeightI"] = "Identity"
        elif (not inp_is_bipolar) and (not wt_is_bipolar):
            ret["TSrcI"] = "Slice<%s>" % inp_hls_str
            ret["TWeightI"] = "Identity"

        # fill in TDstI
        ret["TDstI"] = "Slice<%s>" % out_hls_str

        return ret

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

    def minimize_accumulator_width(self, model):
        weights = model.get_initializer(self.onnx_node.input[1])
        idt = self.get_input_datatype()
        # calculate minimum and maximum values of accumulator
        (acc_min, acc_max) = calculate_matvec_accumulator_range(weights, idt)

        if acc_min < 0:
            if abs(acc_min) > acc_max:
                adt = DataType.get_smallest_possible(acc_min)
            else:
                adt = DataType.get_smallest_possible(0 - acc_max)
        else:
            adt = DataType.get_smallest_possible(acc_max)
        # ensure a datatype divisible by 8-bits in case this is the last node
        bw = roundup_to_integer_multiple(adt.bitwidth(), 8)
        new_adt_name = adt.name.replace(str(adt.bitwidth()), str(bw))
        adt = DataType[new_adt_name]
        self.set_nodeattr("accDataType", adt.name)
        # for no-activation nodes, output dt = acc dt
        self.set_nodeattr("outputDataType", adt.name)

        return DataType[self.get_nodeattr("accDataType")]

    def make_weight_file(self, weights, weight_file_mode, weight_file_name):
        """Produce a file containing given weights in appropriate format for this
        layer. This file can be used for either synthesis or run-time reconfig
        of weights.

        Arguments:
        * weights : numpy array with weights to be put into the file
        * weight_file_mode : one of {hls_header, decoupled_verilog_dat,
          decoupled_runtime}
        * weight_file_name : filename for the weight file to be generated
        """
        # convert weights into hlslib-compatible format
        weight_tensor = self.get_hls_compatible_weight_tensor(weights)
        export_wdt = self.get_weight_datatype()
        # we have converted bipolar weights to binary for export,
        # so use it as such for weight generation
        if self.get_weight_datatype() == DataType.BIPOLAR:
            export_wdt = DataType.BINARY

        elif "decoupled" in weight_file_mode:
            # create a weight stream for various flavors of decoupled mode:
            # transpose weight tensor from (1, PE, WMEM, SIMD) to (1, WMEM, PE, SIMD)
            weight_tensor_unflipped = np.transpose(weight_tensor, (0, 2, 1, 3))

            # reverse SIMD flip for saving weights in .npy
            weight_tensor_simd_flipped = np.flip(weight_tensor_unflipped, axis=-1)
            # PE flip for saving weights in .dat
            weight_tensor_pe_flipped = np.flip(weight_tensor_unflipped, axis=-2)

            # reshape weight tensor (simd_flipped and pe_flipped) to desired shape
            pe = self.get_nodeattr("PE")
            simd = self.get_nodeattr("SIMD")
            # simd_flipped
            weight_tensor_simd_flipped = weight_tensor_simd_flipped.reshape(
                1, -1, pe * simd
            )
            weight_tensor_simd_flipped = weight_tensor_simd_flipped.copy()
            # flipped
            weight_tensor_pe_flipped = weight_tensor_pe_flipped.reshape(
                1, -1, pe * simd
            )
            weight_tensor_pe_flipped = weight_tensor_pe_flipped.copy()
            weight_tensor_pe_flipped2=weight_tensor_pe_flipped
            if weight_file_mode == "decoupled_verilog_dat":
                code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
                # convert weight values into hexstring
                weight_width = self.get_weightstream_width()
                weight_width_padded = roundup_to_integer_multiple(weight_width, 4)
                weight_tensor_pe_flipped = pack_innermost_dim_as_hex_string(
                    weight_tensor_pe_flipped, export_wdt, weight_width_padded, prefix=""
                )
                # add zeroes to pad out file to 1024 entries
                weight_stream = weight_tensor_pe_flipped.flatten()
                weight_stream = weight_stream.copy()
                with open(weight_file_name, "w") as f:
                    for val in weight_stream:
                        f.write(val + "\n")

    def generate_params(self, model, path):
        mem_mode = self.get_nodeattr("mem_mode")
        code_gen_dir = path
        # weights, if not external
        weights = model.get_initializer(self.onnx_node.input[1])

        if mem_mode == "decoupled":
            # save weights as Verilog .dat file
            weight_filename_rtl = "{}/memblock_0.dat".format(code_gen_dir)
            ram_style = self.get_nodeattr("ram_style")
            if ram_style == "ultra":
                # UltraRAM must have no memory initializer, or only zeroes
                # otherwise BRAM will be inferred instead of URAM
                # as a workaround we provide a zero-weight init here
                # TODO handle this in Verilog with an if statement
                weights = np.zeros_like(weights)

            self.make_weight_file(
                weights, "decoupled_verilog_dat", weight_filename_rtl
            )


    def execute_node(self, context, graph):
        pass

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "weights.hpp"']
        self.code_gen_dict["$GLOBALS$"] += ['#include "activations.hpp"']
        self.code_gen_dict["$GLOBALS$"] += ['#include "slidingwindow.h"']
        self.code_gen_dict["$GLOBALS$"] += ['#include "mvau.hpp"']
        self.code_gen_dict["$GLOBALS$"] += ['#include "mvau_pe_fine.hpp"']

    def ipgen_extra_directives(self):
        "Use the extra tcl directives for HLS synthesis to include the extra hpp."
        d = os.path.dirname(sys.modules["finn.custom_op.experimental"].__file__)
        d = os.path.join(d, "../../../../hlslib_extensions")
        return [
            """add_files $config_hwsrcdir/top_%s.cpp -cflags \"-std=c++0x -I%s -I$config_bnnlibdir\""""
             % (self.onnx_node.name, d)
        ]

    def defines(self, var):
        mem_mode = self.get_nodeattr("mem_mode")
        numInputVectors = list(self.get_nodeattr("numInputVectors"))
        numReps = np.prod(numInputVectors)

        pe = self.get_nodeattr("PE")
        mh = self.get_nodeattr("MH")
        mh = mh // pe
        pe = 1
        self.code_gen_dict["$DEFINES$"] = [
            """#define MW1 {}\n #define MH1 {}\n
            #define SIMD1 {}\n #define PE1 {}\n #define WMEM1 {}\n
            \n #define numReps {}""".format(
                self.get_nodeattr("MW"),
                mh,
                self.get_nodeattr("SIMD"),
                pe,
                self.calc_wmem(),
                numReps//self.get_nodeattr("MMV"),
            )
        ]
        self.code_gen_dict["$AP_INT_MAX_W$"] = [str(self.get_ap_int_max_w()//pe)]

        wdt = self.get_weight_datatype()
        self.code_gen_dict["$DEFINES$"].append(
            "#define WP1 {}\n".format(wdt.bitwidth())
        )

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
        # note: the innermost dim is reversed for the input
        self.code_gen_dict["$READNPYDATA$"].append(
            'npy2apintstream<%s, %s, %d, %s>("%s", in0, false);'
            % (packed_hls_type, elem_hls_type, elem_bits, npy_type, npy_in)
        )

        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "decoupled" or mem_mode == "external":
            wdt = self.get_weight_datatype()
            elem_bits = wdt.bitwidth()
            packed_bits = self.get_weightstream_width()
            packed_hls_type = "ap_uint<%d>" % packed_bits
            elem_hls_type = wdt.get_hls_datatype_str()
            npy_type = "float"
            npy_in = "%s/weights.npy" % code_gen_dir

            self.code_gen_dict["$READNPYDATA$"].append(
                'npy2apintstream<%s, %s, %d, %s>("%s", weights, false, numReps);'
                % (packed_hls_type, elem_hls_type, elem_bits, npy_type, npy_in)
            )

    def strm_decl(self):
        mem_mode = self.get_nodeattr("mem_mode")
        self.code_gen_dict["$STREAMDECLARATIONS$"] = []
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> in0 ("in0");'.format(self.get_instream_width())
        )
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> out ("out");'.format(self.get_outstream_width())
        )

        if mem_mode == "decoupled" or mem_mode == "external":
            self.code_gen_dict["$STREAMDECLARATIONS$"].append(
                'hls::stream<ap_uint<{}>> weights ("weights");'.format(
                    self.get_weightstream_width()
                )
            )

    def docompute(self):
        mem_mode = self.get_nodeattr("mem_mode")
        map_to_hls_mult_style = {
            "auto": "ap_resource_dflt()",
            "lut": "ap_resource_lut()",
            "dsp": "ap_resource_dsp()",
        }
        tmpl_args = self.get_template_param_values()
        odtype_hls_str = self.get_output_datatype().get_hls_datatype_str()
        threshs = "PassThroughActivation<%s>()" % odtype_hls_str

        wdt = self.get_weight_datatype()
        if wdt == DataType.BIPOLAR:
            export_wdt = DataType.BINARY
        else:
            export_wdt = wdt
        wdtype_hls_str = export_wdt.get_hls_datatype_str()

        self.code_gen_dict["$DOCOMPUTE$"] = [
            """Matrix_Vector_PE_Batch<MW1, MH1, SIMD1, PE1, {}, {}, {}, {} >
            (in0, out, weights, numReps, {});""".format(
                tmpl_args["TSrcI"],
                tmpl_args["TDstI"],
                tmpl_args["TWeightI"],
                wdtype_hls_str,

                map_to_hls_mult_style[self.get_nodeattr("resType")],
            )
        ]  

    def get_input_datatype(self):
        """Returns FINN DataType of input."""
        return DataType[self.get_nodeattr("inputDataType")]

    def get_weight_datatype(self):
        """Returns FINN DataType of input."""
        return DataType[self.get_nodeattr("weightDataType")]

    def get_output_datatype(self):
        """Returns FINN DataType of output."""
        return DataType[self.get_nodeattr("outputDataType")]


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

    def save_as_npy(self):
        self.code_gen_dict["$SAVEASCNPY$"] = []

    def blackboxfunction(self):                   
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            """void {}(
                hls::stream<ap_uint<{}>> &in0,
                hls::stream<ap_uint<{}>> &weights,
                hls::stream<ap_uint<{}>> &out
                )""".format(
                self.onnx_node.name,
                self.get_instream_width(),
                # divide weight
                (self.get_weightstream_width() // self.get_nodeattr("PE")),
                # divide output
                (self.get_outstream_width() // self.get_nodeattr("PE")),
            )
        ]

    def pragmas(self):
        mem_mode = self.get_nodeattr("mem_mode")

        self.code_gen_dict["$PRAGMAS$"] = ["#pragma HLS INTERFACE axis register off port=in0"]
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis register both port=out")
        in_fifo_depth = self.get_nodeattr("inFIFODepth")
        out_fifo_depth = self.get_nodeattr("outFIFODepth")
        # insert depth pragmas only if specified
        if in_fifo_depth != 0:
            self.code_gen_dict["$PRAGMAS$"].append(
                "#pragma HLS stream depth=%d variable=in0" % in_fifo_depth
            )
        if out_fifo_depth != 0:
            self.code_gen_dict["$PRAGMAS$"].append(
                "#pragma HLS stream depth=%d variable=out" % out_fifo_depth
            )
        self.code_gen_dict["$PRAGMAS$"].append(
            "#pragma HLS INTERFACE ap_ctrl_none port=return"
        )
        self.code_gen_dict["$PRAGMAS$"].append(
            "#pragma HLS INTERFACE axis register off port=weights"
        )
        self.code_gen_dict["$PRAGMAS$"].append(
            "#pragma HLS stream depth=8 variable=weights"
        )    

    def code_generation_ipi(self):
        cmd = []

        pe = self.get_nodeattr("PE")
        mmv = self.get_nodeattr("MMV")
        simd = self.get_nodeattr("SIMD")
        neuron_fold = int(self.get_nodeattr("MH") // pe)
        synapse_fold = int(self.get_nodeattr("MW") // simd)
        wp = self.get_weight_datatype().bitwidth()
        node_name = self.onnx_node.name

        # create a hierarchy for this layer, with the same port names
        clk_name = self.get_verilog_top_module_intf_names()["clk"][0]
        rst_name = self.get_verilog_top_module_intf_names()["rst"][0]
        dout_name = self.get_verilog_top_module_intf_names()["m_axis"][0][0]
        din_name = self.get_verilog_top_module_intf_names()["s_axis"][0][0]
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

        # WEIGHTS
        wwidth=self.get_weightstream_width()
        wwidth_padded=roundup_to_integer_multiple(wwidth, 8)

        # weight transport subhierarchy
        cmd += axis_gather_bcast_scatter("weight_transport", 1, mmv, pe, wwidth, parent_hier=node_name)
        #connect it to input/clk/rst
        cmd.append(
            "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/weight_transport/aclk]"
            % (node_name, clk_name, node_name)
        )
        cmd.append(
            "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/weight_transport/aresetn]"
            % (node_name, rst_name, node_name)
        )

        # add streamer if needed and make connections
        mem_mode = self.get_nodeattr("mem_mode")

        if mem_mode == "external":
            # don't instantiate anything, just connect weight transport to weights input
            w_name = self.get_verilog_top_module_intf_names()["s_axis"][1][0]
            cmd.append(
                "create_bd_intf_pin -mode Slave "
                "-vlnv xilinx.com:interface:axis_rtl:1.0 /%s/%s" % (node_name, w_name)
            )
            cmd.append("connect_bd_intf_net [get_bd_intf_pins %s/%s] "
                        "[get_bd_intf_pins %s/weight_transport/s_0_axis]"
                        % (node_name, w_name, node_name)
            )
        else:
            # instantiate a streamer (or constant) and connect it to the HLS IP
            strm_vlnv = "xilinx.com:user:memstream:1.0"
            strm_inst = "weight_streamer"

            # TODO: instantiate constant when wmem == 1

            mem_init = self.get_nodeattr("code_gen_dir_ipgen") + "/"
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
                    self.calc_wmem(),
                    wwidth_padded,
                    mem_init,
                    self.get_nodeattr("ram_style"),
                    self.calc_wmem(),
                    wwidth_padded,
                    node_name,
                    strm_inst
                )
            )

            # connect output of streamer to input of weight broadcaster
            cmd.append("connect_bd_intf_net [get_bd_intf_pins %s/%s/m_axis_0] "
                        "[get_bd_intf_pins %s/weight_transport/s_0_axis]"
                        % (node_name, strm_inst, node_name)
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

        # INPUTS
        iwidth = self.get_instream_width_padded()
        is_vvau = True if self.get_nodeattr("VVAU") == 1 else False

        # instantiate input buffer(s) and transport tree(s)
        if is_vvau:
            #instantiate a splitter
            cmd += axis_gather_bcast_scatter("vvau_act_transport", 1, mmv, pe, iwidth, parent_hier=node_name)
            #connect it to input/clk/rst
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/vvau_act_transport/aclk]"
                % (node_name, clk_name, node_name)
            )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/vvau_act_transport/aresetn]"
                % (node_name, rst_name, node_name)
            )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/vvau_act_transport/s_0_axis]"
                % (node_name, din_name, node_name)
            )
        else:
            # instantiate a splitter to break the stream into MMV chunks if needed
            if mmv>1:
                cmd += axis_gather_bcast_scatter("mvau_immv_transport", 1, mmv, 1, iwidth, parent_hier=node_name)
                #connect it to input/clk/rst
                cmd.append(
                    "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/mvau_immv_transport/aclk]"
                    % (node_name, clk_name, node_name)
                )
                cmd.append(
                    "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/mvau_immv_transport/aresetn]"
                    % (node_name, rst_name, node_name)
                )
                cmd.append(
                    "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/mvau_immv_transport/s_0_axis]"
                    % (node_name, din_name, node_name)
                )
            # instantiate MMV inputbuffers and broadcast networks
            for m in range(mmv):
                cmd.append("create_bd_cell -type ip -vlnv xilinx.com:user:inputbuf:1.0 %s/inputbuf_%d" % (node_name, m))
                cmd.append("set_property -dict [list CONFIG.WIDTH {%d} CONFIG.DEPTH {%d} CONFIG.NFOLDS {%d} CONFIG.RAM_STYLE {%s}] [get_bd_cells %s/inputbuf_%d]" % (self.get_instream_width_padded(), synapse_fold, neuron_fold, self.get_nodeattr("ibuf_ram_style"), node_name, m))
                # connect inputbuf clk/rst
                cmd.append(
                    "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/inputbuf_%d/aclk]"
                    % (node_name, clk_name, node_name, m)
                )
                cmd.append(
                    "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/inputbuf_%d/aresetn]"
                    % (node_name, rst_name, node_name, m)
                )
                if mmv==1:
                    cmd.append(
                        "connect_bd_intf_net [get_bd_intf_pins %s/%s] [get_bd_intf_pins %s/inputbuf_%d/s_axis]"
                        % (node_name, din_name, node_name, m)
                    )
                else:
                    cmd.append(
                        "connect_bd_intf_net [get_bd_intf_pins %s/mvau_immv_transport/m_0_%s_axis] [get_bd_intf_pins %s/inputbuf_%d/s_axis]"
                        % (node_name, din_name, node_name, m)
                    )
                # instantiate a bcast network
                cmd += axis_gather_bcast_scatter("mvau_act_transport_"+str(m), 1, pe, 1, iwidth//mmv, parent_hier=node_name)
                #connect it to input/clk/rst
                cmd.append(
                    "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/mvau_act_transport_%d/aclk]"
                    % (node_name, clk_name, node_name, m)
                )
                cmd.append(
                    "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/mvau_act_transport_%d/aresetn]"
                    % (node_name, rst_name, node_name, m)
                )
                cmd.append(
                    "connect_bd_intf_net [get_bd_intf_pins %s/inputbuf_%d/m_axis] [get_bd_intf_pins %s/mvau_act_transport_%d/s_0_axis]"
                    % (node_name, m, node_name, m)
                )

        # OUTPUTS
        # accumulator gather network
        accwidth = DataType[self.get_nodeattr("accDataType")].bitwidth()
        cmd += axis_gather_bcast_scatter("acc_transport", pe*mmv, 1, 1, accwidth, parent_hier=node_name)
        cmd.append(
            "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/acc_transport/aclk]"
            % (node_name, clk_name, node_name)
        )
        cmd.append(
            "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/acc_transport/aresetn]"
            % (node_name, rst_name, node_name)
        )
        # TODO: implement MMV terminator with axis switch

        # PEs
        for m in range(mmv):   
            # instantiate the hls ip "pe" number of times
            for i in range(pe):
                cmd.append(
                    "create_bd_cell -type ip -vlnv %s /%s/PE_%d_%d"
                    % (self.get_nodeattr("ip_vlnv"), node_name, m, i)
                )
                cmd.append(
                    "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/PE_%d_%d/%s]"
                    % (node_name, clk_name, node_name, m, i, clk_name)
                )
                cmd.append(
                    "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/PE_%d_%d/%s]"
                    % (node_name, rst_name, node_name, m, i, rst_name)
                )
                # connect transport/accumulator network(s) to PEs
                if is_vvau:
                    cmd.append(
                        "connect_bd_intf_net [get_bd_intf_pins %s/vvau_act_transport/m_%d_%d_axis] "
                        "[get_bd_intf_pins %s/PE_%d_%d/%s]"
                        % (node_name, m, i, node_name, m, i, din_name)
                    )
                else:
                    cmd.append(
                        "connect_bd_intf_net [get_bd_intf_pins %s/mvau_act_transport_%d/m_%d_0_axis] "
                        "[get_bd_intf_pins %s/PE_%d_%d/%s]"
                        % (node_name, m, i, node_name, m, i, din_name)
                    )
                cmd.append(
                    "connect_bd_intf_net [get_bd_intf_pins %s/acc_transport/s_%d_axis] "
                    "[get_bd_intf_pins %s/PE_%d_%d/%s]"
                    % (node_name, m*pe+i, node_name, m, i, dout_name)
                )
                cmd.append(
                    "connect_bd_intf_net [get_bd_intf_pins %s/weight_transport/m_%d_%d_axis] "
                    "[get_bd_intf_pins %s/PE_%d_%d/weights_V_V]"
                    % (node_name, m, i, node_name, m, i)
                )

        cmd.append(
            "connect_bd_intf_net [get_bd_intf_pins %s/acc_transport/m_0_0_axis] "
            "[get_bd_intf_pins %s/%s]"
            % (node_name, node_name, dout_name)
        )

        cmd.append("save_bd_design")
        return cmd

    def get_verilog_top_module_intf_names(self):
        intf_names = super().get_verilog_top_module_intf_names()
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "external":
            intf_names["s_axis"] = ["in0_V_V", "weights_V_V"]
        return intf_names

    def infer_node_datatype(self):
        pass
    
    def make_shape_compatible_op(self):
        pass

    def verify_node(self):
        pass
