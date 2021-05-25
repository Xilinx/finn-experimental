
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
import os
import numpy as np
import subprocess
from onnx import TensorProto, helper
from finn.core.datatype import DataType
from finn.custom_op.fpgadataflow.streamingfclayer_batch import StreamingFCLayer_Batch
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
from . import templates
import textwrap

# ONNX i/o tensor shape assumptions for StreamingFCLayer:
# input 0 is the input tensor, shape (.., i_size) = (..., MW)
# input 1 is the weight tensor, shape (i_size, o_size) = (MW, MH)
# (optional) input 2 is the thresholds tensor, shape (o_size, n_thres)
# output 0 is the output tensor, shape (.., o_size) = (..., MH)
# the ... here can be any shape (representing groups of vectors)

class StreamingFCLayer_MMV_FG_Batch(StreamingFCLayer_Batch):
    """Class that corresponds to finn-hls StreamingFCLayer_Batch function."""

    def __init__(self, onnx_node):
        super().__init__(onnx_node)
        self.decoupled_wrapper = templates.decoupled_wrapper

    def get_nodeattr_types(self):
        my_attrs = {
            "ibuf_ram_style": (
                "s",
                False,
                "auto",
                {"auto", "block", "distributed", "ultra"},
            ),
            "fine_grained" : ("i", False, 0, {0, 1}),
            "MMV" : ("i", False, 1),
    
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

    def calc_tmem(self):
        """Calculates and returns TMEM."""
        if self.get_nodeattr("noActivation") == 1:
            return 0
        else:
            mh = self.get_nodeattr("MH")
            pe = self.get_nodeattr("PE")
            return mh // pe


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
        if (mmode == "decoupled" and mstyle != "ultra") or (
            mmode == "const" and self.calc_wmem() <= 128
        ):
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
        if (mmode == "decoupled" and mstyle in ["distributed", "ultra"]) or (
            mmode == "const" and self.calc_wmem() <= 128
        ):
            return 0
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
        if (mmode == "decoupled" and mstyle == "distributed") or (
            mmode == "const" and self.calc_wmem() <= 128
        ):
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
        noact = self.get_nodeattr("noActivation")
        if noact == 0:
            odt = self.get_output_datatype()
            B = odt.bitwidth()
            thr_luts = (2 ** B - 1) * acc_bits * math.ceil(self.calc_tmem() / 64)
            comp_luts = (2 ** B - 1) * acc_bits

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
        if (
            self.get_nodeattr("mem_mode") == "decoupled"
            or self.get_nodeattr("mem_mode") == "external"
        ):
            pe = self.get_nodeattr("PE")
            simd = self.get_nodeattr("SIMD")
            wp = self.get_weight_datatype().bitwidth()
            w_width = pe * simd * wp
            return w_width
        else:
            return 0

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
        if self.get_nodeattr("fine_grained")==True:

           weightstream = self.get_weightstream_width()//self.get_nodeattr("PE")
        else:
           weightstream = self.get_weightstream_width()
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
        if len(self.onnx_node.input) > 2:
            thresholds = model.get_initializer(self.onnx_node.input[2])
        else:
            thresholds = None
        idt = self.get_input_datatype()
        # calculate minimum and maximum values of accumulator
        (acc_min, acc_max) = calculate_matvec_accumulator_range(weights, idt)
        if thresholds is not None:
            threshold_tensor = self.get_hls_compatible_threshold_tensor(thresholds)
            # set threshold datatype (and accumulator datatype implicitly)
            min_threshold = thresholds.min()
            max_threshold = thresholds.max()
            # clip threshold values
            clip_upper = None
            clip_lower = None
            if max_threshold > acc_max + 1:
                clip_upper = acc_max + 1
            if min_threshold < acc_min:
                clip_lower = acc_min
            if (clip_lower is not None) or (clip_upper is not None):
                warnings.warn("Clipping some thresholds in %s" % self.onnx_node.name)
                thresholds = np.clip(thresholds, clip_lower, clip_upper)
                model.set_initializer(self.onnx_node.input[2], thresholds)
                threshold_tensor = self.get_hls_compatible_threshold_tensor(thresholds)
                min_threshold = thresholds.min()
                max_threshold = thresholds.max()
            # get range required by threshold values
            tdt_min = min(acc_min, min_threshold)
            tdt_max = max(acc_max, max_threshold)
            if tdt_min < 0:
                if abs(tdt_min) > tdt_max:
                    tdt = DataType.get_smallest_possible(tdt_min)
                else:
                    tdt = DataType.get_smallest_possible(0 - tdt_max)
            else:
                tdt = DataType.get_smallest_possible(tdt_max)
            assert np.vectorize(tdt.allowed)(threshold_tensor).all(), (
                "Thresholds in %s can't be expressed with type %s"
                % (self.onnx_node.name, str(tdt))
            )
            self.set_nodeattr("accDataType", tdt.name)
        else:
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
        inp_is_bipolar = self.get_input_datatype() == DataType.BIPOLAR
        wt_is_bipolar = self.get_weight_datatype() == DataType.BIPOLAR
        # reinterpret inp/wt as bipolar if bin_xnor_mode is iset
        inp_is_binary = self.get_input_datatype() == DataType.BINARY
        wt_is_binary = self.get_weight_datatype() == DataType.BINARY
        bin_xnor_mode = self.get_nodeattr("binaryXnorMode") == 1
        inp_is_bipolar = inp_is_bipolar or (inp_is_binary and bin_xnor_mode)
        wt_is_bipolar = wt_is_bipolar or (wt_is_binary and bin_xnor_mode)
        if inp_is_bipolar and wt_is_bipolar:
            # ensure all thresholds are nonnegative
            assert (orig_thres_matrix >= 0).all()
            # ensure all thresholds are integer
            assert (orig_thres_matrix.astype(np.int32) == orig_thres_matrix).all()
        ret = orig_thres_matrix
        # workaround for vivado_hls threshold bug
        if ret[0][0] == 0:
            ret = np.copy(ret)
            ret[0][0] = 1
            warnings.warn(
                "Setting 0-valued first threshold to 1 to avoid vivado_hls bug"
            )
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
        if weight_file_mode == "hls_header":
            weight_hls_code = numpy_to_hls_code(
                weight_tensor, export_wdt, "weights", True, True
            )
            # write weights into C++ header file as dictated by finn-hlslib
            f_weights = open(weight_file_name, "w")
            if export_wdt.bitwidth() != 1:
                f_weights.write(
                    "const FixedPointWeights<{},{},{},{}> weights = ".format(
                        self.get_nodeattr("SIMD"),
                        export_wdt.get_hls_datatype_str(),
                        self.get_nodeattr("PE"),
                        self.calc_wmem(),
                    )
                )
            else:
                f_weights.write(
                    "const BinaryWeights<{},{},{}> weights = ".format(
                        self.get_nodeattr("SIMD"),
                        self.get_nodeattr("PE"),
                        self.calc_wmem(),
                    )
                )
            f_weights.write(weight_hls_code)
            f_weights.close()
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
            if weight_file_mode == "decoupled_npy":
                # save weight stream into npy for cppsim
                np.save(weight_file_name, weight_tensor_simd_flipped)
            elif weight_file_mode == "decoupled_verilog_dat":
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

                ##
                if self.get_weightstream_width() > 40000:
                    # TODO create a function for finding a suitable wtrm factor
                    wstrm_factor = 2
                    split_weightstream_width=self.get_weightstream_width()//wstrm_factor
                    split_weightstream_width_padded=roundup_to_integer_multiple(split_weightstream_width, 4)

                    assert pe%wstrm_factor==0 ,"number of weightstreams must be divisible by pe"

                    split_array=np.split(weight_tensor_pe_flipped2, wstrm_factor, axis=-1)
                    split_array.reverse()
                    head, tail=os.path.split(self.get_nodeattr("code_gen_dir_ipgen"))
                    for w in range(wstrm_factor):
                        split_array_hex = pack_innermost_dim_as_hex_string(
                           split_array[w], export_wdt, split_weightstream_width_padded, prefix=""
                        )
                        code_gen_dir_name =  os.environ["FINN_BUILD_DIR"] + "/" + tail + "/wstrm" + str(w)
                        if not os.path.isdir(code_gen_dir_name):
                           os.makedirs(code_gen_dir_name)
                        
                        weight_file_name = code_gen_dir_name + "/memblock_0.dat"
                        split_stream = split_array_hex.flatten()
                        split_stream = split_stream.copy()
                        with open(weight_file_name, "w") as f:
                          for val in split_stream:
                              f.write(val + "\n")
                              
                else:
                    wstrm_factor = 1
                    
                        
            elif weight_file_mode == "decoupled_runtime":
                # memstream axi-lite interface will map each mem line to
                # one or multiple 32-bit words
                weight_width = self.get_weightstream_width()
                words_per_memwidth = 2 ** math.ceil(math.log2(weight_width / 32))
                if words_per_memwidth < 1:
                    words_per_memwidth = 1
                weight_width_padded = words_per_memwidth * 32
                # first, pack and ensure padding to 32 bits
                weight_tensor_pe_flipped = pack_innermost_dim_as_hex_string(
                    weight_tensor_pe_flipped, export_wdt, weight_width_padded, prefix=""
                )
                weight_stream = weight_tensor_pe_flipped.flatten()
                weight_stream = weight_stream.copy()
                with open(weight_file_name, "w") as f:
                    for val in weight_stream:
                        # split into groups of 8 hex digits (= 32 bits)
                        words_32b = textwrap.wrap(val, 8)
                        words_32b.reverse()
                        for word_32b in words_32b:
                            f.write(word_32b + "\n")
            else:
                raise Exception("Unknown weight_file_mode")

        else:
            raise Exception("Unknown weight_file_mode")

    def generate_params(self, model, path):
        mem_mode = self.get_nodeattr("mem_mode")
        code_gen_dir = path
        # weights, if not external
        weights = model.get_initializer(self.onnx_node.input[1])

        if mem_mode == "const":
            # save hlslib-compatible weights in params.h
            weight_filename = "{}/params.h".format(code_gen_dir)
            self.make_weight_file(weights, "hls_header", weight_filename)
        elif mem_mode == "decoupled" or mem_mode == "external":
            weight_filename_sim = "{}/weights.npy".format(code_gen_dir)
            # save decoupled weights for cppsim
            self.make_weight_file(weights, "decoupled_npy", weight_filename_sim)
            if mem_mode == "decoupled":
                # also save weights as Verilog .dat file
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
        else:
            raise Exception(
                """Please set mem_mode to "const", "decoupled", or "external",
                currently no other parameter value is supported!"""
            )

        # save thresholds in thresh.h
        if len(self.onnx_node.input) > 2:
            thresholds = model.get_initializer(self.onnx_node.input[2])
            if thresholds is not None:
                threshold_tensor = self.get_hls_compatible_threshold_tensor(thresholds)
                # use UINT32 threshold export for bipolar times bipolar
                inp_is_bipolar = self.get_input_datatype() == DataType.BIPOLAR
                wt_is_bipolar = self.get_weight_datatype() == DataType.BIPOLAR
                # reinterpret inp/wt as bipolar if bin_xnor_mode is iset
                inp_is_binary = self.get_input_datatype() == DataType.BINARY
                wt_is_binary = self.get_weight_datatype() == DataType.BINARY
                bin_xnor_mode = self.get_nodeattr("binaryXnorMode") == 1
                inp_is_bipolar = inp_is_bipolar or (inp_is_binary and bin_xnor_mode)
                wt_is_bipolar = wt_is_bipolar or (wt_is_binary and bin_xnor_mode)
                # get computed threshold datatype from attribute
                tdt = DataType[self.get_nodeattr("accDataType")]

                assert np.vectorize(tdt.allowed)(threshold_tensor).all(), (
                    "Thresholds in %s can't be expressed with type %s"
                    % (self.onnx_node.name, str(tdt))
                )
                thresholds_hls_code = numpy_to_hls_code(
                    threshold_tensor, tdt, "thresholds", False, True
                )
                # write thresholds into thresh.h
                f_thresh = open("{}/thresh.h".format(code_gen_dir), "w")
                tdt_hls = tdt.get_hls_datatype_str()
                # use binary to export bipolar activations
                export_odt = self.get_output_datatype()
                if self.get_output_datatype() == DataType.BIPOLAR:
                    export_odt = DataType.BINARY
                odt_hls = export_odt.get_hls_datatype_str()
                f_thresh.write(
                    "static ThresholdsActivation<{},{},{},{},{},{},{}> threshs \
                    = ".format(
                        self.calc_tmem(),
                        self.get_nodeattr("PE"),
                        threshold_tensor.shape[-1],
                        tdt_hls,
                        odt_hls,
                        self.get_nodeattr("ActVal"),
                        "comp::less_equal<%s>" % tdt_hls,
                    )
                )
                f_thresh.write(thresholds_hls_code)
                f_thresh.close()


    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        mem_mode = self.get_nodeattr("mem_mode")
        node = self.onnx_node

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

        # create a npy file fore each input of the node (in_ind is input index)
        in_ind = 0
        for inputs in node.input:
            # it is assumed that the first input of the node is the data input
            # the second input are the weights
            # the third input are the thresholds
            if in_ind == 0:
                assert (
                    str(context[inputs].dtype) == "float32"
                ), """Input datatype is
                not float32 as expected."""
                expected_inp_shape = self.get_folded_input_shape()
                reshaped_input = context[inputs].reshape(expected_inp_shape)
                if self.get_input_datatype() == DataType.BIPOLAR:
                    # store bipolar activations as binary
                    reshaped_input = (reshaped_input + 1) / 2
                    export_idt = DataType.BINARY
                else:
                    export_idt = self.get_input_datatype()
                # make copy before saving the array
                reshaped_input = reshaped_input.copy()
                np.save(
                    os.path.join(code_gen_dir, "input_{}.npy".format(in_ind)),
                    reshaped_input,
                )
            elif in_ind > 2:
                raise Exception("Unexpected input found for StreamingFCLayer")
            in_ind += 1

        if mode == "cppsim":
            # execute the precompiled model
            super().exec_precompiled_singlenode_model()
            # load output npy file
            super().npy_to_dynamic_output(context)
            # reinterpret binary output as bipolar where needed
            if self.get_output_datatype() == DataType.BIPOLAR:
                out = context[node.output[0]]
                out = 2 * out - 1
                context[node.output[0]] = out
            assert (
                context[node.output[0]].shape == self.get_folded_output_shape()
            ), """Output shape is not as expected"""
            # reshape output to have expected shape
            oshape = self.get_normal_output_shape()
            context[node.output[0]] = context[node.output[0]].reshape(*oshape)
        elif mode == "rtlsim":
            sim = self.get_rtlsim()
            nbits = self.get_instream_width()
            inp = npy_to_rtlsim_input(
                "{}/input_0.npy".format(code_gen_dir), export_idt, nbits
            )
            super().reset_rtlsim(sim)
            super().toggle_clk(sim)
            if mem_mode == "external" or mem_mode == "decoupled":
                wnbits = self.get_weightstream_width()
                export_wdt = self.get_weight_datatype()
                # we have converted bipolar weights to binary for export,
                # so use it as such for weight generation
                if self.get_weight_datatype() == DataType.BIPOLAR:
                    export_wdt = DataType.BINARY
                wei = npy_to_rtlsim_input(
                    "{}/weights.npy".format(code_gen_dir), export_wdt, wnbits
                )
                num_w_reps = np.prod(self.get_nodeattr("numInputVectors"))
                io_dict = {
                    "inputs": {"in0": inp, "weights": wei * num_w_reps},
                    "outputs": {"out": []},
                }
                self.rtlsim_multi_io(sim, io_dict)
                output = io_dict["outputs"]["out"]
            else:
                output = self.rtlsim(sim, inp)
            odt = self.get_output_datatype()
            target_bits = odt.bitwidth()
            packed_bits = self.get_outstream_width()
            out_npy_path = "{}/output.npy".format(code_gen_dir)
            out_shape = self.get_folded_output_shape()
            rtlsim_output_to_npy(
                output, out_npy_path, odt, out_shape, packed_bits, target_bits
            )

            # load and reshape output
            output = np.load(out_npy_path)
            oshape = self.get_normal_output_shape()
            output = np.asarray([output], dtype=np.float32).reshape(*oshape)
            context[node.output[0]] = output
        else:
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following value ("cppsim", "rtlsim")""".format(
                    mode
                )
            )

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "weights.hpp"']
        self.code_gen_dict["$GLOBALS$"] += ['#include "activations.hpp"']

        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "const":
            # self.code_gen_dict["$GLOBALS$"] += ['#include "params.h"']
            pass
        elif self.get_nodeattr("fine_grained")==True:
            self.code_gen_dict["$GLOBALS$"] += ['#include "slidingwindow.h"']
            self.code_gen_dict["$GLOBALS$"] += ['#include "mvau.hpp"']
        elif mem_mode == "decoupled" or mem_mode == "external":
            self.code_gen_dict["$GLOBALS$"] += ['#include "mvau.hpp"']
        else:
            raise Exception(
                """Please set mem_mode to "const", "decoupled", or "external",
                currently no other parameter value is supported!"""
            )
        if self.calc_tmem() != 0:
            # TODO find a better way of checking for no pregenerated thresholds
            self.code_gen_dict["$GLOBALS$"] += ['#include "thresh.h"']

    def defines(self, var):
        mem_mode = self.get_nodeattr("mem_mode")
        numInputVectors = list(self.get_nodeattr("numInputVectors"))
        numReps = np.prod(numInputVectors)
        if mem_mode == "const" or self.get_nodeattr("noActivation") != 1:
          self.code_gen_dict["$DEFINES$"] = [
             """#define MW1 {}\n #define MH1 {}\n
             #define SIMD1 {}\n #define PE1 {}\n #define WMEM1 {}\n
             #define TMEM1 {}\n #define numReps {}""".format(
                 self.get_nodeattr("MW"),
                 self.get_nodeattr("MH"),
                 self.get_nodeattr("SIMD"),
                 self.get_nodeattr("PE"),
                 self.calc_wmem(),
                 self.calc_tmem(),
                 numReps,
             )
          ]
        if self.get_nodeattr("fine_grained") == 1:
          assert(self.get_nodeattr("mem_mode") != "const"), """mem_mode must be constant for fine grained"""
          assert(self.get_nodeattr("noActivation") == 1), """noActivation must be 1 for fine grained"""

          pe = self.get_nodeattr("PE")
          mh = self.get_nodeattr("MH")
          mh = mh // pe
          pe = 1
          self.code_gen_dict["$DEFINES$"] = [
              """#define MW1 {}\n #define MH1 {}\n
             #define SIMD1 {}\n #define PE1 {}\n #define WMEM1 {}\n
             #define TMEM1 {}\n #define numReps {}""".format(
                  self.get_nodeattr("MW"),
                  mh,
                  self.get_nodeattr("SIMD"),
                  pe,
                  self.calc_wmem(),
                  self.calc_tmem(),
                  numReps//self.get_nodeattr("MMV"),
              )
          ]
          self.code_gen_dict["$AP_INT_MAX_W$"] = [str(self.get_ap_int_max_w()//pe)]
        elif mem_mode == "decoupled" or mem_mode == "external":
          pe = self.get_nodeattr("PE")
          mh = self.get_nodeattr("MH")
          self.code_gen_dict["$DEFINES$"] = [
              """#define MW1 {}\n #define MH1 {}\n
             #define SIMD1 {}\n #define PE1 {}\n #define WMEM1 {}\n
             #define TMEM1 {}\n #define numReps {}""".format(
                  self.get_nodeattr("MW"),
                  mh,
                  self.get_nodeattr("SIMD"),
                  pe,
                  self.calc_wmem(),
                  self.calc_tmem(),
                  numReps,
              )
          ]
          self.code_gen_dict["$AP_INT_MAX_W$"] = [str(self.get_ap_int_max_w())]
        if mem_mode == "decoupled" or mem_mode == "external":
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
        if self.calc_tmem() == 0:
            odtype_hls_str = self.get_output_datatype().get_hls_datatype_str()
            threshs = "PassThroughActivation<%s>()" % odtype_hls_str
        else:
            threshs = "threshs"
        if mem_mode == "const":
            node = self.onnx_node
            self.code_gen_dict["$DOCOMPUTE$"] = [
                """{}<MW1, MH1, SIMD1, PE1, {}, {}, {}>
                (in0, out, weights, {}, numReps, {});""".format(
                    node.op_type,
                    tmpl_args["TSrcI"],
                    tmpl_args["TDstI"],
                    tmpl_args["TWeightI"],
                    threshs,
                    map_to_hls_mult_style[self.get_nodeattr("resType")],
                )
            ]

        elif self.get_nodeattr("fine_grained") == True:
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
                     

        elif mem_mode == "decoupled" or mem_mode == "external":
            wdt = self.get_weight_datatype()
            if wdt == DataType.BIPOLAR:
                export_wdt = DataType.BINARY
            else:
                export_wdt = wdt
            wdtype_hls_str = export_wdt.get_hls_datatype_str()
            self.code_gen_dict["$DOCOMPUTE$"] = [
                """Matrix_Vector_Activate_Stream_Batch<MW1, MH1, SIMD1, PE1, {}, {}, {}, {} >
                (in0, out, weights, {}, numReps, {});""".format(
                    tmpl_args["TSrcI"],
                    tmpl_args["TDstI"],
                    tmpl_args["TWeightI"],
                    wdtype_hls_str,
                    threshs,
                    map_to_hls_mult_style[self.get_nodeattr("resType")],
                )
            ]

        else:
            raise Exception(
                """Please set mem_mode to "const", "decoupled", or "external",
                currently no other parameter value is supported!"""
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

    def save_as_npy(self):
        self.code_gen_dict["$SAVEASCNPY$"] = []

    def blackboxfunction(self):
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "const":
            self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
                """void {}(hls::stream<ap_uint<{}>> &in0,
                    hls::stream<ap_uint<{}>> &out
                    )""".format(
                    self.onnx_node.name,
                    self.get_instream_width(),
                    self.get_outstream_width(),
                )
            ]
        #elif mem_mode != "const" and self.get_nodeattr("noActivation") == 1:                      
        elif self.get_nodeattr("fine_grained") == 1: 
            assert(self.get_nodeattr("mem_mode") != "const"), """mem_mode must be constant for fine grained"""
            assert(self.get_nodeattr("noActivation") == 1), """noActivation must be 1 for fine grained""" 
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
   
        elif mem_mode == "decoupled" or mem_mode == "external":
            
            self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
                """void {}(
                    hls::stream<ap_uint<{}>> &in0,
                    hls::stream<ap_uint<{}>> &weights,
                    hls::stream<ap_uint<{}>> &out
                    )""".format(
                    self.onnx_node.name,
                    self.get_instream_width(),
                    # divide weight
                    self.get_weightstream_width(),
                    self.get_outstream_width(),
                )
            ]

        else:
            raise Exception(
                """Please set mem_mode to "const" or "decoupled", currently no other
                    parameter value is supported!"""
            )

    def pragmas(self):
        mem_mode = self.get_nodeattr("mem_mode")
        if self.get_nodeattr("fine_grained") == True:
           self.code_gen_dict["$PRAGMAS$"] = ["#pragma HLS INTERFACE axis register off port=in0"]
           self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis register both port=out")
        else:
           self.code_gen_dict["$PRAGMAS$"] = ["#pragma HLS INTERFACE axis port=in0"]
           self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=out")
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

        if mem_mode == "const":
            self.code_gen_dict["$PRAGMAS$"].append('#include "params.h"')
            # the weight tensor is ap_uint<simd*prec> [PE][WMEM]
            # partition for parallel access along the PE dimension (dim 1)
            self.code_gen_dict["$PRAGMAS$"].append(
                (
                    "#pragma HLS ARRAY_PARTITION variable=weights.m_weights "
                    "complete dim=1"
                )
            )
        elif mem_mode == "decoupled" or mem_mode == "external":
            if self.get_nodeattr("fine_grained") == True:
               self.code_gen_dict["$PRAGMAS$"].append(
                  "#pragma HLS INTERFACE axis register off port=weights"
               )
            else:
               self.code_gen_dict["$PRAGMAS$"].append(
                  "#pragma HLS INTERFACE axis port=weights"
               )
            self.code_gen_dict["$PRAGMAS$"].append(
                "#pragma HLS stream depth=8 variable=weights"
            )

        else:
            raise Exception(
                """Please set mem_mode to "const", "decoupled", or external,
                currently no other parameter value is supported!"""
            )

        # the threshold tensor is acc_type [PE][TMEM][N_THRES]
        # partition for parallel access along PE and N_THRES
        # dimensions (dims 1 and 3)
        if self.calc_tmem() != 0:
            # TODO find a better way of checking for no pregenerated thresholds
            self.code_gen_dict["$PRAGMAS$"].append(
                (
                    "#pragma HLS ARRAY_PARTITION variable=threshs.m_thresholds "
                    "complete dim=1"
                )
            )
            self.code_gen_dict["$PRAGMAS$"].append(
                (
                    "#pragma HLS ARRAY_PARTITION variable=threshs.m_thresholds "
                    "complete dim=3"
                )
            )
    
    def weight_clk_and_reset(self, cmd, w, mmv):
        node_name = self.onnx_node.name
        clk_name = self.get_verilog_top_module_intf_names()["clk"][0]
        rst_name = self.get_verilog_top_module_intf_names()["rst"][0]
        dout_name = self.get_verilog_top_module_intf_names()["m_axis"][0]
        din_name = self.get_verilog_top_module_intf_names()["s_axis"][0]
        strm_inst = node_name + "_wstrm"
        # 2. weight broadcaster reset and clock
        if mmv == 0:
           cmd.append(
             "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/axis_broadcaster_weight_mmv_%d/aresetn]"
             % (node_name, rst_name, node_name, w)
           )
           cmd.append(
             "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/axis_broadcaster_weight_mmv_%d/aclk]"
             % (node_name, clk_name, node_name, w)
           )
 
           #streamer reset and clock
           cmd.append(
             "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s_%d/aresetn]"
             % (node_name, rst_name, node_name, strm_inst, w)
           )
           cmd.append(
             "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s_%d/aclk]"
             % (node_name, clk_name, node_name, strm_inst, w)
           ) 

        # connect clk and reset - weight_splitter
        cmd.append(
           "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/axis_weight_splitter_%d_%d/aresetn]"
           % (node_name, rst_name, node_name, mmv, w)
        )
        cmd.append(
           "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/axis_weight_splitter_%d_%d/aclk]"
           % (node_name, clk_name, node_name, mmv, w)
        )
        return cmd

    def instantiate_initial_mmv_blocks(self, cmd):
       mmv_value = self.get_nodeattr("MMV")
       node_name = self.onnx_node.name
       clk_name = self.get_verilog_top_module_intf_names()["clk"][0]
       rst_name = self.get_verilog_top_module_intf_names()["rst"][0]
       dout_name = self.get_verilog_top_module_intf_names()["m_axis"][0]
       din_name = self.get_verilog_top_module_intf_names()["s_axis"][0]
       # split inputs to mmv broadcaster
       cmd.append(
          "create_bd_cell -type ip -vlnv user.org:user:axis_split_core:1.0 %s/axis_splitter_inputs_mmv" 
           % (node_name)
       )
       cmd.append(
           "set_property -dict [list CONFIG.S_AXIS_TDATA_WIDTH_PAD {%d} \
                                     CONFIG.C_AXIS_TDATA_WIDTH {%d} \
                                     CONFIG.M_AXIS_TDATA_WIDTH_PAD {%d} \
                                     CONFIG.C_NUM_MI_SLOTS {%d} \
            ]\
            [get_bd_cells %s/axis_splitter_inputs_mmv]" 
            % (roundup_to_integer_multiple(self.get_instream_width() * mmv_value, 8), 
               self.get_instream_width() * mmv_value, 
               self.get_instream_width(), 
               mmv_value, 
               node_name)
       ) 

       # clock and reset axis splitter

       cmd.append(
          "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/axis_splitter_inputs_mmv/aresetn]"
           % (node_name, rst_name, node_name)
       )
       cmd.append(
          "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/axis_splitter_inputs_mmv/aclk]"
          % (node_name, clk_name, node_name)
       )

       # connect input of block to input splitter mmv
       cmd.append(
          "connect_bd_intf_net [get_bd_intf_pins %s/%s] [get_bd_intf_pins %s/axis_splitter_inputs_mmv/s_axis]" 
          % (node_name, din_name, node_name)
       )

       if self.get_nodeattr("MMV") > 1:
          # create combiner before giving it to the final interconnect
          cmd.append(
             "create_bd_cell -type ip -vlnv user.org:user:axis_combiner_v1_1_19_top:1.0 %s/axis_combiner_mmv" 
              % (node_name)
          )
          cmd.append(
             "set_property -dict [list CONFIG.C_AXIS_TDATA_WIDTH {%d} \
                                       CONFIG.C_AXIS_SIGNAL_SET {0x00000003} \
                                       CONFIG.C_NUM_SI_SLOTS {%d} \
                                 ] \
                                 [get_bd_cells %s/axis_combiner_mmv]" 
             % (self.get_outstream_width(), mmv_value, node_name)
          ) 

          ## instantiate axis stream switch
          #cmd.append(
          #   "create_bd_cell -type ip -vlnv xilinx.com:ip:axis_switch:1.1 %s/axis_switch_mmv"
          #    % (node_name)
          #)
          #cmd.append(
          #   "set_property -dict [list CONFIG.NUM_SI {%d} \
          #                                     CONFIG.NUM_MI {1} \
          #                                     CONFIG.TDATA_NUM_BYTES {%d} \
          #                                     CONFIG.ARB_ON_MAX_XFERS {%d} \
          #                                     CONFIG.ARB_ALGORITHM {1} ] \
          #                               [get_bd_cells %s/axis_switch_mmv]" 
          #   % (mmv_value, self.get_outstream_width()//8, self.get_nodeattr("MH")//self.get_nodeattr("PE"), node_name)
          #)

          cmd.append(
             "create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 %s/xlconstant_mmv" 
             % (node_name)
          )
          cmd.append(
             "set_property -dict [list CONFIG.CONST_WIDTH {%d} CONFIG.CONST_VAL {0}] [get_bd_cells %s/xlconstant_mmv]"
              % (mmv_value, node_name)
          )
          #cmd.append(
          #   "connect_bd_net [get_bd_pins %s/xlconstant_mmv/dout] [get_bd_pins %s/axis_switch_mmv/s_req_suppress]" 
          #   % (node_name, node_name)
          #)
          #cmd.append(
          #   "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/axis_switch_mmv/aclk]" 
          #   % (node_name, clk_name, node_name)
          #)
          #cmd.append(
          #   "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/axis_switch_mmv/aresetn]" 
          #   % (node_name, rst_name, node_name)
          #)
          # 3. output combiner
          cmd.append(
             "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/axis_combiner_mmv/aresetn]"
             % (node_name, rst_name, node_name)
          )
          cmd.append(
             "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/axis_combiner_mmv/aclk]"
             % (node_name, clk_name, node_name)
          ) 
       return cmd       

    def code_generation_ipi(self):
        cmd = []
        # add streamer if needed
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "decoupled":
            runtime_writable = self.get_nodeattr("runtime_writeable_weights") == 1

            pe = self.get_nodeattr("PE")
            neuron_fold = int(self.get_nodeattr("MH") // self.get_nodeattr("PE"))
            synapse_fold = int(self.get_nodeattr("MW") // self.get_nodeattr("SIMD"))
            simd = self.get_nodeattr("SIMD")
            wp = self.get_weight_datatype().bitwidth()

            if self.get_nodeattr("ram_style") == "ultra":
                assert (
                    runtime_writable == 1
                ), "Layer with URAM weights must have runtime_writeable_weights=1"
            node_name = self.onnx_node.name

            # create a hierarchy for this layer, with the same port names
            clk_name = self.get_verilog_top_module_intf_names()["clk"][0]
            rst_name = self.get_verilog_top_module_intf_names()["rst"][0]
            dout_name = self.get_verilog_top_module_intf_names()["m_axis"][0]
            din_name = self.get_verilog_top_module_intf_names()["s_axis"][0]
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


            if self.get_nodeattr("fine_grained")==True:
               mmv_value = self.get_nodeattr("MMV")
               cmd = self.instantiate_initial_mmv_blocks(cmd)

               for mmv in range(mmv_value):   
                  assert (self.get_nodeattr("mem_mode")!="const"), "mem_mode must be constant for fine_grained"
                  assert (self.get_nodeattr("noActivation")==1), "noActivation must be 1 for fine_grained"
                  # instantiate the hls ip "pe" number of times
                  for i in range(pe):
                      cmd.append(
                         "create_bd_cell -type ip -vlnv %s /%s/%s_%d_%d"
                         % (self.get_nodeattr("ip_vlnv"), node_name, node_name, mmv, i)
                      )
                  # WEIGHTS 

                  # Condition for split wstrms or constant blocks
                  if self.calc_wmem() != 1: 
                     if self.get_weightstream_width() > 40000:
                        wstrm_factor = 2
                     else:
                        wstrm_factor = 1

                     split_weightstream_width=self.get_weightstream_width()//wstrm_factor
                     split_weightstream_width_padded=roundup_to_integer_multiple(split_weightstream_width, 8)

                     for w in range(wstrm_factor):
                        if mmv == 0:
                           if wstrm_factor == 1: 
                              mem_init = self.get_nodeattr("code_gen_dir_ipgen") + "/"
                           else:
                              mem_init = self.get_nodeattr("code_gen_dir_ipgen") + "/wstrm" + str(w) + "/"
                           cmd.append(
                             "create_bd_cell -type ip -vlnv %s /%s/%s_%d"
                             % (strm_vlnv, node_name, strm_inst, w)
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
                             "] [get_bd_cells /%s/%s_%d]"
                             % (
                                self.calc_wmem(),
                                split_weightstream_width_padded,
                                mem_init,
                                self.get_nodeattr("ram_style"),
                                self.calc_wmem(),
                                split_weightstream_width_padded,
                                node_name,
                                strm_inst,
                                w
                              )
                           )

                           # broadcast weights to mmv splitters
                           cmd.append("create_bd_cell -type ip -vlnv user.org:user:extend_broadcaster2:1.0 %s/axis_broadcaster_weight_mmv_%d" % (node_name, w))
                           cmd.append("set_property -dict [list CONFIG.C_AXIS_TDATA_WIDTH {%d} CONFIG.C_NUM_MI_SLOTS {%s}] [get_bd_cells %s/axis_broadcaster_weight_mmv_%d]" % (split_weightstream_width_padded, mmv_value, node_name, w))


                           # connect output of streamer to input of weight broadcaster
                           cmd.append("connect_bd_intf_net [get_bd_intf_pins %s/%s_%s/m_axis_0] "
                                      "[get_bd_intf_pins %s/axis_broadcaster_weight_mmv_%d/s_axis]"
                                      % (node_name, strm_inst, w, node_name, w)
                           )

                        #instantiate and configure the weight splitter
                        cmd.append("create_bd_cell -type ip -vlnv user.org:user:axis_split_core:1.0 %s/axis_weight_splitter_%d_%d" % (node_name, mmv, w))
                        cmd.append("set_property -dict [list CONFIG.S_AXIS_TDATA_WIDTH_PAD {%d} CONFIG.C_AXIS_TDATA_WIDTH {%d} CONFIG.M_AXIS_TDATA_WIDTH_PAD {%d} CONFIG.C_NUM_MI_SLOTS {%d}] [get_bd_cells %s/axis_weight_splitter_%d_%d]" % (split_weightstream_width_padded, split_weightstream_width, self.get_weight_splitter_output_width_padded(), pe//wstrm_factor, node_name, mmv, w))

                        cmd = self.weight_clk_and_reset(cmd, w, mmv)
 
                        #connect output of weight broadcaster to input of weight splitter
                        cmd.append("connect_bd_intf_net [get_bd_intf_pins %s/axis_broadcaster_weight_mmv_%d/m_axis_%02d] [get_bd_intf_pins %s/axis_weight_splitter_%d_%d/s_axis]" % (node_name, w, mmv, node_name, mmv, w))

 
                        for i in range(pe//wstrm_factor):
                           cmd.append(
                             "connect_bd_intf_net [get_bd_intf_pins %s/axis_weight_splitter_%d_%d/m_axis_%02d] "
                             "[get_bd_intf_pins %s/%s_%d_%d/weights_V_V]"
                             % (node_name, mmv, w, i, node_name, node_name, mmv, w * (pe//wstrm_factor) + i)
                             )    




                  else:
                     dat_file = self.get_nodeattr("code_gen_dir_ipgen") + "/memblock_0.dat" 
                     df = open(dat_file, "r")
                     for i in range(pe):
                         cmd.append("create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 %s/xlconstant_data_%d_%02d" % (node_name, mmv, i))
                         df.seek((pe - 1 - i)*((simd*wp)//4), 0)
                         weight_val = df.read((simd*wp)//4)
                         cmd.append("set_property -dict [list CONFIG.CONST_WIDTH {%d} CONFIG.CONST_VAL {%s}] [get_bd_cells %s/xlconstant_data_%d_%02d]" % (self.get_weight_splitter_output_width_padded(), '0x' + weight_val, node_name, mmv, i))

                         cmd.append("connect_bd_net [get_bd_pins %s/xlconstant_data_%d_%02d/dout] [get_bd_pins %s/%s_%d_%d/weights_V_V_TDATA]" % (node_name, mmv, i, node_name, node_name, mmv, i))

                         cmd.append("create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 %s/xlconstant_valid_%d_%02d" % (node_name, mmv, i))

                         cmd.append("set_property -dict [list CONFIG.CONST_WIDTH {%d} CONFIG.CONST_VAL {%s}] [get_bd_cells %s/xlconstant_valid_%d_%02d]" % (1, 1, node_name, mmv, i))
 
                         cmd.append("connect_bd_net [get_bd_pins %s/xlconstant_valid_%d_%02d/dout] [get_bd_pins %s/%s_%d_%d/weights_V_V_TVALID]" % (node_name, mmv, i, node_name, node_name, mmv, i))
                        
                         cmd.append("save_bd_design")
                     df.close()

                  # instantiate combiner block and set input parameters
                  cmd.append("create_bd_cell -type ip -vlnv user.org:user:axis_combiner_v1_1_19_top:1.0 %s/axis_combiner_output_%d" % (node_name, mmv))
                  cmd.append("set_property -dict [list CONFIG.C_AXIS_TDATA_WIDTH {%d} CONFIG.C_AXIS_SIGNAL_SET {0x00000003} CONFIG.C_NUM_SI_SLOTS {%d}] [get_bd_cells %s/axis_combiner_output_%d]" % (self.get_outstream_width() // self.get_nodeattr("PE"), pe, node_name, mmv)) 
                 
                  # INPUTS
                  # instantiate input buffer
                  cmd.append("create_bd_cell -type ip -vlnv user.org:user:inputbuf:1.0 %s/inputbuf_%d" % (node_name, mmv))
                  cmd.append("set_property -dict [list CONFIG.WIDTH {%d} CONFIG.DEPTH {%d} CONFIG.NFOLDS {%d} CONFIG.RAM_STYLE {%s}] [get_bd_cells %s/inputbuf_%d]" % (self.get_instream_width_padded(), synapse_fold, neuron_fold, self.get_nodeattr("ibuf_ram_style"), node_name, mmv))
                 
                  # instantiate input broadcaster and set number of masters
                  cmd.append("create_bd_cell -type ip -vlnv user.org:user:extend_broadcaster2:1.0 %s/axis_broadcaster_input_%d" % (node_name, mmv))
                  cmd.append("set_property -dict [list CONFIG.C_AXIS_TDATA_WIDTH {%d} CONFIG.C_NUM_MI_SLOTS {%s}] [get_bd_cells %s/axis_broadcaster_input_%d]" % (self.get_instream_width_padded(), pe, node_name, mmv))

                  # connect input splitter outputs to input buffer
                  cmd.append("connect_bd_intf_net [get_bd_intf_pins %s/axis_splitter_inputs_mmv/m_axis_%02d] " 
                             "[get_bd_intf_pins %s/inputbuf_%d/s_axis]" 
                             % (node_name, mmv, node_name, mmv
                             )
                             )

                  # connect inputbuf to input of each broadcaster
                  cmd.append(
                     "connect_bd_intf_net [get_bd_intf_pins %s/inputbuf_%d/m_axis] "
                     "[get_bd_intf_pins %s/axis_broadcaster_input_%d/s_axis]"
                     % (node_name, mmv, node_name, mmv)
                  )

                  # connect broadcasters to pe
                  for i in range(pe):
                       cmd.append(
                           "connect_bd_intf_net [get_bd_intf_pins %s/axis_broadcaster_input_%d/m_axis_%02d] "
                           "[get_bd_intf_pins %s/%s_%d_%d/%s]"
                           % (node_name, mmv, i, node_name, node_name, mmv, i, din_name)
                       ) 




                  for i in range(pe):
                       cmd.append("connect_bd_intf_net [get_bd_intf_pins %s/%s_%d_%d/out_V_V] [get_bd_intf_pins %s/axis_combiner_output_%d/s_axis_%02d]" % (node_name, node_name, mmv, i, node_name, mmv, i))


                  if mmv_value > 1:
                     # connect output of combiner to synchronising combiner 
                     cmd.append(
                        "connect_bd_intf_net [get_bd_intf_pins %s/axis_combiner_output_%d/m_axis] "
                        "[get_bd_intf_pins %s/axis_combiner_mmv/s_axis_%02d]" 
                        % (node_name, mmv, node_name, mmv)
                     )

                     
                     if mmv == 0:
                        cmd.append(
                           "connect_bd_intf_net [get_bd_intf_pins %s/axis_combiner_mmv/m_axis] "
                           "[get_bd_intf_pins %s/%s] " 
                           % (node_name, node_name, dout_name)
                        )

                  else:
                     # connect output of combiner to block output
                     cmd.append("connect_bd_intf_net [get_bd_intf_pins %s/axis_combiner_output_%d/m_axis] [get_bd_intf_pins %s/%s]" % (node_name, mmv, node_name, dout_name))                       
   

                  for i in range(pe):
                     cmd.append(
                       "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s_%d_%d/%s]"
                       % (node_name, rst_name, node_name, node_name, mmv, i, rst_name)
                     )
                     cmd.append(
                      "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s_%d_%d/%s]"
                      % (node_name, clk_name, node_name, node_name, mmv, i, clk_name)
                     )
  
                  # connect clk and reset - combiner
                  cmd.append(
                    "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/axis_combiner_output_%d/aresetn]"
                    % (node_name, rst_name, node_name, mmv)
                  )
                  cmd.append(
                    "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/axis_combiner_output_%d/aclk]"
                     % (node_name, clk_name, node_name, mmv)
                  ) 
  
                  cmd.append(
                    "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/inputbuf_%d/aresetn]"
                    % (node_name, rst_name, node_name, mmv)
                  )
                  cmd.append(
                    "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/inputbuf_%d/aclk]"
                     % (node_name, clk_name, node_name, mmv)
                  ) 
                  # connect clk and reset - input broadcaster
                  cmd.append(
                    "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/axis_broadcaster_input_%d/aresetn]"
                    % (node_name, rst_name, node_name, mmv)
                  )
                  cmd.append(
                    "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/axis_broadcaster_input_%d/aclk]"
                    % (node_name, clk_name, node_name, mmv)
                  ) 


            cmd.append("save_bd_design")
        elif mem_mode == "const":
            # base class impl sufficient for const mode
            return super().code_generation_ipi()
        else:
            raise Exception("Unrecognized mem_mode for StreamingFCLayer")
        return cmd

    def get_verilog_top_module_intf_names(self):
        intf_names = super().get_verilog_top_module_intf_names()
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "external":
            intf_names["s_axis"] = ["in0_V_V", "weights_V_V"]
        if mem_mode == "decoupled":
            # only expose axilite interface if attribute is set
            runtime_writable = self.get_nodeattr("runtime_writeable_weights") == 1
            if runtime_writable:
                intf_names["axilite"] = ["s_axilite"]
        return intf_names

