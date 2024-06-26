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
from functools import partial

from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.analysis.fpgadataflow.res_estimation import res_estimation
from finn.transformation.fpgadataflow.set_folding import SetFolding
from qonnx.transformation.base import Transformation

from finnexperimental.transformation.fpgadataflow.set_mem_mode import SetMemMode
from finnexperimental.util.platforms import DEFAULT_RES_LIMITS, platforms


class AllocateResources(Transformation):
    """Fold a dataflow design to a target fps within resource constraints."""

    def __init__(self, fps_target, clk_ns, platform, devices=1, limits=DEFAULT_RES_LIMITS):
        super().__init__()
        self.clk_ns = clk_ns
        self.fps_target = fps_target
        self.platform = platform
        self.cpf_target = 1 if (fps_target == -1) else int((10**9 / clk_ns) / fps_target)
        self.max_luts = limits[0] * sum(
            [r["LUT"] for r in platforms[platform](devices).resource_count_dict.values()]
        )
        self.max_bram = limits[2] * sum(
            [r["BRAM_18K"] for r in platforms[platform](devices).resource_count_dict.values()]
        )
        self.max_uram = limits[3] * sum(
            [r["URAM"] for r in platforms[platform](devices).resource_count_dict.values()]
        )
        self.max_dsp = limits[4] * sum(
            [r["DSP"] for r in platforms[platform](devices).resource_count_dict.values()]
        )

    def apply(self, model):
        feasible_implementation = False
        while not feasible_implementation:
            model = model.transform(SetFolding(self.cpf_target))
            exp_cycles_dict = model.analysis(exp_cycles_per_layer)
            achieved_cycles_per_frame = max(exp_cycles_dict.values())

            # if achieved is more than target, it means we can't fold enough;
            # re-fold with target set to the largest cycle delay of all layers,
            # to minimize resource usage
            if self.cpf_target < achieved_cycles_per_frame:
                model = model.transform(SetFolding(achieved_cycles_per_frame))

            # set mem modes and styles, to ensure optimum memory resource utilization
            # TODO
            # determine if the utilization is unbalanced
            # if too many LUTs relative to BRAM, decrease lutmem_thr and allocate again
            # if too many BRAM relative to LUTs, increase lutmem_thr and allocate again
            # increase or decrease in increments of 64
            lutmem_thr = 128
            model = model.transform(SetMemMode(lutmem_thr))

            # do resource estimation, and get relative utilization of the target plaform
            resource_usage = model.analysis(
                partial(res_estimation, fpgapart="xcu250-figd2104-2L-e")
            )
            luts = sum([r["LUT"] for r in resource_usage.values()])
            brams = sum([r["BRAM_18K"] for r in resource_usage.values()])
            urams = sum([r["URAM"] for r in resource_usage.values()])
            dsps = sum([r["DSP"] for r in resource_usage.values()])

            assert not (
                urams > 0 and self.max_uram == 0
            ), "URAMs allocated but target platform has no URAM resources"
            # determine if we're overrunning the available resources; if so, lower
            # cpf target and fold again
            if (
                luts > self.max_luts
                or brams > self.max_bram
                or urams > self.max_uram
                or dsps > self.max_dsp
            ):
                if self.max_uram > 0:
                    self.cpf_target *= max(
                        luts / self.max_luts,
                        brams / self.max_bram,
                        urams / self.max_uram,
                        dsps / self.max_dsp,
                    )
                else:
                    self.cpf_target *= max(
                        luts / self.max_luts,
                        brams / self.max_bram,
                        dsps / self.max_dsp,
                    )
            else:
                feasible_implementation = True

        return (model, False)
