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
from abc import abstractmethod

DC = -1  # explicit value for don't care
RES_UTIL_GUIDE = np.array([.7,.5,.80,.50,.50])

class Platform():

    def __init__(self, nslr=1, ndevices=1, sll_count=[], eth_slr=0, eth_gbps=0):
        self.nslr = nslr
        self.sll_count = sll_count
        self.eth_slr = eth_slr
        self.eth_gbps = eth_gbps
        self.ndevices = ndevices

    @property
    @abstractmethod
    def compute_resources(self):
        pass

    @property
    def guide_resources(self):
        guide = []
        for i in range(self.nslr):
            guide_res = np.array(self.compute_resources[i])*RES_UTIL_GUIDE
            guide.append(list(guide_res.astype(int)))
        return guide

    @property
    def resource_count_dict(self):
        res = dict()
        for i in range(self.nslr*self.ndevices):
            slr_res = dict()
            slr_res["LUT"] = self.compute_resources[i][0]
            slr_res["FF"] = self.compute_resources[i][1]
            slr_res["BRAM_18K"] = self.compute_resources[i][2]
            slr_res["URAM"] = self.compute_resources[i][3]
            slr_res["DSP"] = self.compute_resources[i][4]
            res["slr"+str(i)] = slr_res
        return res

    @property
    def compute_connection_cost(self):
        x = np.full((self.nslr*self.ndevices, self.nslr*self.ndevices), DC)
        # build connection cost matrix for one device's SLRs
        xlocal = np.full((self.nslr, self.nslr), DC)
        for i in range(self.nslr):
            for j in  range(self.nslr):
                if i == j:
                    xlocal[i][j] = 0
                elif abs(i-j) == 1:
                    xlocal[i][j] = 1
        # tile connection cost matrices for entire system
        for i in range(self.ndevices):
            x[i*self.nslr:(i+1)*self.nslr, i*self.nslr:(i+1)*self.nslr] = xlocal
        # set cost for ethernet connections, assuming daisy-chaining
        for i in range(self.ndevices-1):
            x[i*self.nslr+self.eth_slr][(i+1)*self.nslr+self.eth_slr] = 10
            x[(i+1)*self.nslr+self.eth_slr][i*self.nslr+self.eth_slr] = 10
        return x

    @property
    def compute_connection_resource(self):
        sll = np.full((self.nslr*self.ndevices, self.nslr*self.ndevices), 0)
        # build connection resource matrix for one device's SLRs
        slllocal = np.full((self.nslr, self.nslr), -1)
        for i in range(self.nslr):
            for j in  range(self.nslr):
                if i == j:
                    # no SLL constraint when going from one SLR to itself
                    slllocal[i][j] = -1
                else:
                    slllocal[i][j] = self.sll_count[i][j]
        # tile connection cost matrices for entire system
        for i in range(self.ndevices):
            sll[i*self.nslr:(i+1)*self.nslr, i*self.nslr:(i+1)*self.nslr] = slllocal
        # set cost for ethernet connections, assuming daisy-chaining
        eth = np.full((self.nslr*self.ndevices, self.nslr*self.ndevices), 0)
        # no Eth throughput constraints from one SLR to itself
        for i in range(self.ndevices*self.nslr):
            eth[i][i] = -1
        # apply symmetric ETH throughput constraints between the SLRs that have GTXes
        for i in range(self.ndevices-1):
            eth[i*self.nslr+self.eth_slr][(i+1)*self.nslr+self.eth_slr] = self.eth_gbps * (10**9)
            eth[(i+1)*self.nslr+self.eth_slr][i*self.nslr+self.eth_slr] = self.eth_gbps * (10**9)
        # pack sll and eth info in one list-of-list-of-tuple structure
        constraints = []
        for i in range(self.ndevices*self.nslr):
            constraints_line = []
            for j in range(self.ndevices*self.nslr):
                # make sure not to constrain both resources at the same time
                # constrain for Eth throughput between SLRs on different devices
                # constrain for SLLs between SLRs on same device
                is_offchip = (i//self.nslr != j//self.nslr)
                constraints_line.append((-1 if is_offchip else sll[i][j], eth[i][j] if is_offchip else -1))
            constraints.append(constraints_line)
        return constraints

    def map_device_to_slr(self, idx):
        """Given a global SLR index, return device id and local slr index"""
        assert idx <= self.nslr*self.ndevices
        return (idx%self.nslr, idx//self.nslr)


class Zynq7020_Platform(Platform):

    def __init__(self, ndevices=1):
        super(Zynq7020_Platform, self).__init__(nslr=1, ndevices=ndevices, sll_count=[[0]], eth_slr=0, eth_gbps=1)

    @property
    def compute_resources(self):
        # U50 has identical resource counts on both SLRs
        return [[53200, 2*53200, 280, 0, 220] for i in range(2)]


class ZU3EG_Platform(Platform):

    def __init__(self, ndevices=1):
        super(ZU3EG_Platform, self).__init__(nslr=1, ndevices=ndevices, sll_count=[[0]], eth_slr=0, eth_gbps=1)

    @property
    def compute_resources(self):
        # U50 has identical resource counts on both SLRs
        return [[71000, 2*71000, 412, 0, 360] for i in range(2)]


class ZU7EV_Platform(Platform):

    def __init__(self, ndevices=1):
        super(ZU7EV_Platform, self).__init__(nslr=1, ndevices=ndevices, sll_count=[[0]], eth_slr=0, eth_gbps=1)

    @property
    def compute_resources(self):
        # U50 has identical resource counts on both SLRs
        return [[230000, 2*230000, 610, 92, 1728] for i in range(2)]


class Alveo_NxU50_Platform(Platform):

    def __init__(self, ndevices=1):
        sll_counts = [[0, 1662],[1285,0]]
        super(Alveo_NxU50_Platform, self).__init__(nslr=2, ndevices=ndevices, sll_count=sll_counts, eth_slr=1, eth_gbps=100)

    @property
    def compute_resources(self):
        # U50 has identical resource counts on both SLRs
        return [[365000,2*365000,2*564, 304, 2580] for i in range(2)]


class Alveo_NxU200_Platform(Platform):

    def __init__(self, ndevices=1):
        sll_counts = [[0, 1662, 0], [1285, 0, 1450], [0, 1568, 0]]
        super(Alveo_NxU200_Platform, self).__init__(nslr=3, ndevices=ndevices, sll_count=sll_counts, eth_slr=2, eth_gbps=100)

    @property
    def compute_resources(self):
        return [[355000, 723000, 2*638, 320, 2265],
                [160000, 331000, 2*326, 160, 1317],
                [355000, 723000, 2*638, 320, 2265]]


class Alveo_NxU250_Platform(Platform):

    def __init__(self, ndevices=1):
        sll_counts = [[0, 1662, 0, 0], [1285, 0, 0, 1450], [0, 1575, 0, 1568], [0, 0, 1709, 0]]
        super(Alveo_NxU250_Platform, self).__init__(nslr=4, ndevices=ndevices, sll_count=sll_counts, eth_slr=3, eth_gbps=100)

    @property
    def compute_resources(self):
        # U250 has identical resource counts on all 4 SLRs
        return [[345000,2*345000,2*500, 320, 2877] for i in range(4)]


class Alveo_NxU280_Platform(Platform):

    def __init__(self, ndevices=1):
        sll_counts = [[0, 1662, 0], [1285, 0, 1450], [0, 1568, 0]]
        super(Alveo_NxU280_Platform, self).__init__(nslr=3, ndevices=ndevices, sll_count=sll_counts, eth_slr=2, eth_gbps=100)

    @property
    def compute_resources(self):
        return [[369000, 746000, 2*507, 320, 2733],
                [333000, 675000, 2*468, 320, 2877],
                [367000, 729000, 2*512, 320, 2880]]


platforms = dict()
platforms["U50"] = Alveo_NxU50_Platform
platforms["U200"] = Alveo_NxU200_Platform
platforms["U250"] = Alveo_NxU250_Platform
platforms["U280"] = Alveo_NxU280_Platform
platforms["Pynq-Z1"] = Zynq7020_Platform
platforms["Pynq-Z2"] = Zynq7020_Platform
platforms["Ultra96"] = ZU3EG_Platform
platforms["ZCU104"] = ZU7EV_Platform
