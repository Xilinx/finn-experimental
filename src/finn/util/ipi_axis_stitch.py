import math

# create the TCL to instantiate a broadcast-then-scatter tree in a new BD hierarchy
def axis_gather_bcast_scatter(new_hier, njoins, nreplicas, nsplits, ibits, parent_hier=None):
    assert njoins<=256, "njoins supported up to 256"
    assert nreplicas<=256, "nreplicas supported up to 256"
    assert nsplits<=256, "nsplits supported up to 256"
    cmd = []
    if parent_hier is not None:
        hier_name = parent_hier + "/" + new_hier
    else:
        hier_name = new_hier
    # Output video subsystem hierarchy
    cmd.append("create_bd_cell -type hier %s" %(hier_name))

    # Create interface ports (AXI Stream in/out)
    cmd.append("create_bd_pin -dir I -type clk %s/aclk" % (hier_name))
    cmd.append("create_bd_pin -dir I -type rst %s/aresetn" % (hier_name))
    for i in range(njoins):
        cmd.append("create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 %s/s_%d_axis" % (hier_name, i))
    for i in range(nreplicas):
        for j in range(nsplits):
            cmd.append("create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 %s/m_%d_%d_axis" % (hier_name, i, j))

    #corner case nreplicas=1 nsplits=1 njoins=1
    if nreplicas==1 and nsplits==1 and njoins==1:
        cmd.append("connect_bd_intf_net [get_bd_intf_pins %s/s_0_axis] [get_bd_intf_pins %s/m_0_0_axis]" % (hier_name, hier_name))
        return cmd        

    # Instantiate Combiner(s) for joins
    cmd += axis_gather("gather", njoins, ibits, parent_hier=hier_name)
    cmd.append("connect_bd_net [get_bd_pins %s/gather/aclk] [get_bd_pins %s/aclk]" % (hier_name, hier_name))
    cmd.append("connect_bd_net [get_bd_pins %s/gather/aresetn] [get_bd_pins %s/aresetn]" % (hier_name, hier_name))
    for i in range(njoins):
        cmd.append("connect_bd_intf_net [get_bd_intf_pins %s/s_%d_axis] [get_bd_intf_pins %s/gather/s_%d_axis]" % (hier_name, i, hier_name, i))
    
    bcast_ibits = ibits*njoins
    bcast_ibytes = int(math.ceil(bcast_ibits/8))
    cmd += axis_bcast("broadcast", nreplicas, bcast_ibytes, parent_hier=hier_name)
    cmd.append("connect_bd_net [get_bd_pins %s/broadcast/aclk] [get_bd_pins %s/aclk]" % (hier_name, hier_name))
    cmd.append("connect_bd_net [get_bd_pins %s/broadcast/aresetn] [get_bd_pins %s/aresetn]" % (hier_name, hier_name))
    cmd.append("connect_bd_intf_net [get_bd_intf_pins %s/gather/m_axis] [get_bd_intf_pins %s/broadcast/s_axis]" % (hier_name, hier_name))

    # Instantiate Splitter(s) for nsplits
    if nsplits == 1:
        if nreplicas != 1:
            for m in range(nreplicas):
                cmd.append("connect_bd_intf_net [get_bd_intf_pins %s/m_%d_0_axis] [get_bd_intf_pins %s/broadcast/m_%d_axis]" % (hier_name, m, hier_name, m//16, m%16))
        else:
            cmd.append("connect_bd_intf_net [get_bd_intf_pins %s/m_0_0_axis] [get_bd_intf_pins %s/gather/m_axis]" % (hier_name, hier_name))
    else:
        assert bcast_ibits % nsplits == 0
        split_obits = bcast_ibits//nsplits
        for m in range(nreplicas):
            # instantiate splitters for every output of the broadcaster
            cell_name = "scatter_"+str(m)
            cmd += axis_scatter(cell_name, nsplits, split_obits, parent_hier=hier_name)
            cmd.append("connect_bd_net [get_bd_pins %s/%s/aclk] [get_bd_pins %s/aclk]" % (hier_name, cell_name, hier_name))
            cmd.append("connect_bd_net [get_bd_pins %s/%s/aresetn] [get_bd_pins %s/aresetn]" % (hier_name, cell_name, hier_name))
            cmd.append("connect_bd_intf_net [get_bd_intf_pins %s/%s/s_axis] [get_bd_intf_pins %s/broadcast/m_%d_axis]" % (hier_name, cell_name, hier_name, m))
            for i in range(nsplits):
                cmd.append("connect_bd_intf_net [get_bd_intf_pins %s/m_%d_%d_axis] [get_bd_intf_pins %s/%s/m_%d_axis]" % (hier_name, m, i, hier_name, cell_name, i))

    return cmd

# create the TCL to instantiate a gather tree in a new BD hierarchy
def axis_gather(new_hier, njoins, ibits, parent_hier=None):
    ibytes = int(math.ceil(ibits/8))
    assert njoins<=256, "njoins supported up to 256"
    cmd = []
    if parent_hier is not None:
        hier_name = parent_hier + "/" + new_hier
    else:
        hier_name = new_hier
    # Create hierarchy
    cmd.append("create_bd_cell -type hier %s" %(hier_name))

    # Create interface ports (AXI Stream in/out)
    cmd.append("create_bd_pin -dir I -type clk %s/aclk" % (hier_name))
    cmd.append("create_bd_pin -dir I -type rst %s/aresetn" % (hier_name))
    for i in range(njoins):
        cmd.append("create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 %s/s_%d_axis" % (hier_name, i))
    cmd.append("create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 %s/m_axis" % (hier_name))

    #corner case njoins=1 straight-through connection
    if njoins==1:
        cmd.append("connect_bd_intf_net [get_bd_intf_pins %s/s_0_axis] [get_bd_intf_pins %s/m_axis]" % (hier_name, hier_name))
        return cmd        
    else:
        # Instantiate Combiner(s) for joins
        join_hierarchical = True if njoins>16 else False
        join_ncomb = int(math.ceil(njoins/16))
        #instantiate leaf combiner(s)
        for i in range(join_ncomb):
            num_si = min(16, njoins - i*16)
            cell_name = "%s/join_comb_%d" % (hier_name, i)
            if num_si > 1:
                cmd.append("create_bd_cell -type ip -vlnv xilinx.com:ip:axis_combiner:1.1 %s" % (cell_name))
                cmd.append("set_property -dict [list CONFIG.TDATA_NUM_BYTES.VALUE_SRC USER CONFIG.TDATA_NUM_BYTES {%d} CONFIG.NUM_SI {%d}] [get_bd_cells %s]" % (ibytes, num_si, cell_name))
                for j in range(num_si):
                    cmd.append("connect_bd_intf_net [get_bd_intf_pins %s/s_%d_axis] [get_bd_intf_pins %s/S%02d_AXIS]" % (hier_name, 16*i+j, cell_name, j))
            else:
                # the only way we got here is if we have njoins=N*16+1 (N>0)
                # we need to convert the tdata of this one remaining stream to 16*ibytes
                cmd.append("create_bd_cell -type ip -vlnv xilinx.com:ip:axis_subset_converter:1.1 %s" % (cell_name))
                cmd.append("set_property -dict [list CONFIG.M_TDATA_NUM_BYTES.VALUE_SRC USER CONFIG.S_TDATA_NUM_BYTES.VALUE_SRC USER] [get_bd_cells %s]" %(cell_name))
                cmd.append("set_property -dict [list CONFIG.S_TDATA_NUM_BYTES {%d} CONFIG.M_TDATA_NUM_BYTES {%d}] [get_bd_cells %s]" %(ibytes, 16*ibytes, cell_name))
                cmd.append("connect_bd_intf_net [get_bd_intf_pins %s/S_AXIS] [get_bd_intf_pins %s/s_%d_axis]" % (cell_name, hier_name, 16*i))       
            cmd.append("connect_bd_net [get_bd_pins %s/aclk] [get_bd_pins %s/aclk]" % (hier_name, cell_name))
            cmd.append("connect_bd_net [get_bd_pins %s/aresetn] [get_bd_pins %s/aresetn]" % (hier_name, cell_name))


        # instantiate root combiner if needed
        if join_hierarchical:
            cell_name = "%s/join_root_comb" % (hier_name)
            cmd.append("create_bd_cell -type ip -vlnv xilinx.com:ip:axis_combiner:1.1 %s" % (cell_name))
            cmd.append("set_property -dict [list CONFIG.TDATA_NUM_BYTES.VALUE_SRC USER CONFIG.TDATA_NUM_BYTES {%d} CONFIG.NUM_SI {%d}] [get_bd_cells %s]" % (16*ibytes, join_ncomb, cell_name))
            cmd.append("connect_bd_net [get_bd_pins %s/aclk] [get_bd_pins %s/aclk]" % (hier_name, cell_name))
            cmd.append("connect_bd_net [get_bd_pins %s/aresetn] [get_bd_pins %s/aresetn]" % (hier_name, cell_name))
            for i in range(join_ncomb):
                cmd.append("connect_bd_intf_net [get_bd_intf_pins %s/join_comb_%d/M_AXIS] [get_bd_intf_pins %s/S%02d_AXIS]" % (hier_name, i, cell_name, i))

        # instantiate substream converter if needed to realign data and chop off padding
        if (join_hierarchical and njoins%16 != 0) or ibits%8 != 0:
            comb_ibytes = min(16,njoins)*ibytes*join_ncomb
            comb_obytes = int(math.ceil(njoins*ibits/8))
            padding_bits = 8*comb_obytes - njoins*ibits
            remap_string = []
            for j in range(njoins):
                bit_offset = j*ibytes*8
                remap_string.append("tdata[%d:%d]" % (bit_offset+ibits-1, bit_offset))
            if padding_bits > 0:
                remap_string.append("%d'b" %(padding_bits) + "0"*padding_bits)
            remap_string = ','.join(reversed(remap_string))
        else:
            # no combiner required, connect and return
            cmd.append("connect_bd_intf_net [get_bd_intf_pins %s/%s/M_AXIS] [get_bd_intf_pins %s/m_axis]" % (hier_name, "join_root_comb" if join_hierarchical else "join_comb_0", hier_name))
            return cmd

        cell_name = "%s/join_root_conv" % (hier_name)
        cmd.append("create_bd_cell -type ip -vlnv xilinx.com:ip:axis_subset_converter:1.1 %s" % (cell_name))
        cmd.append("set_property -dict [list CONFIG.M_TDATA_NUM_BYTES.VALUE_SRC USER CONFIG.S_TDATA_NUM_BYTES.VALUE_SRC USER] [get_bd_cells %s]" %(cell_name))
        cmd.append("set_property -dict [list CONFIG.S_TDATA_NUM_BYTES {%d} CONFIG.M_TDATA_NUM_BYTES {%d} CONFIG.TDATA_REMAP {%s}] [get_bd_cells %s]" %(comb_ibytes, comb_obytes, remap_string, cell_name))
        cmd.append("connect_bd_net [get_bd_pins %s/aclk] [get_bd_pins %s/aclk]" % (hier_name, cell_name))
        cmd.append("connect_bd_net [get_bd_pins %s/aresetn] [get_bd_pins %s/aresetn]" % (hier_name, cell_name))
        if join_hierarchical:
            cmd.append("connect_bd_intf_net [get_bd_intf_pins %s/S_AXIS] [get_bd_intf_pins %s/join_root_comb/M_AXIS]" % (cell_name, hier_name))       
        else:
            cmd.append("connect_bd_intf_net [get_bd_intf_pins %s/S_AXIS] [get_bd_intf_pins %s/join_comb_0/M_AXIS]" % (cell_name, hier_name))       
        cmd.append("connect_bd_intf_net [get_bd_intf_pins %s/m_axis] [get_bd_intf_pins %s/M_AXIS]" % (hier_name, cell_name))       

    return cmd

def axis_bcast(new_hier, nreplicas, ibytes, parent_hier=None):
    assert nreplicas<=256, "nreplicas supported up to 256"
    cmd = []
    if parent_hier is not None:
        hier_name = parent_hier + "/" + new_hier
    else:
        hier_name = new_hier
    # Create hierarchy
    cmd.append("create_bd_cell -type hier %s" %(hier_name))

    # Create interface ports (AXI Stream in/out)
    cmd.append("create_bd_pin -dir I -type clk %s/aclk" % (hier_name))
    cmd.append("create_bd_pin -dir I -type rst %s/aresetn" % (hier_name))
    for i in range(nreplicas):
        cmd.append("create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 %s/m_%d_axis" % (hier_name, i))
    cmd.append("create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 %s/s_axis" % (hier_name))

    # Instantiate Broadcaster(s) for nreplicas
    if nreplicas==1:
        cmd.append("connect_bd_intf_net [get_bd_intf_pins %s/s_axis] [get_bd_intf_pins %s/m_0_axis]" % (hier_name, hier_name))
    else:
        rep_hierarchical = True if nreplicas>16 else False
        rep_nbcast = int(math.ceil(nreplicas/16))
        #instantiate root broadcaster if nreplicas > 16
        if rep_hierarchical:
            cell_name = "%s/rep_root_bcast" % (hier_name)
            cmd.append("create_bd_cell -type ip -vlnv xilinx.com:ip:axis_broadcaster:1.1 %s" % (cell_name))
            cmd.append("set_property -dict [list CONFIG.M_TDATA_NUM_BYTES.VALUE_SRC USER CONFIG.S_TDATA_NUM_BYTES.VALUE_SRC USER CONFIG.NUM_MI {%d}] [get_bd_cells %s]" % (rep_nbcast, cell_name))
            cmd.append("set_property -dict [list CONFIG.M_TDATA_NUM_BYTES {%d} CONFIG.S_TDATA_NUM_BYTES {%d}] [get_bd_cells %s]" % (ibytes, ibytes, cell_name))
            cmd.append("connect_bd_net [get_bd_pins %s/aclk] [get_bd_pins %s/aclk]" % (hier_name, cell_name))
            cmd.append("connect_bd_net [get_bd_pins %s/aresetn] [get_bd_pins %s/aresetn]" % (hier_name, cell_name))
            cmd.append("connect_bd_intf_net [get_bd_intf_pins %s/s_axis] [get_bd_intf_pins %s/S_AXIS]" % (hier_name, cell_name))
        #instantiate leaf broadcaster(s)
        for i in range(rep_nbcast):
            num_mi = min(16, nreplicas - i*16)
            if num_mi>1:
                cell_name = "%s/rep_bcast_%d" % (hier_name, i)
                cmd.append("create_bd_cell -type ip -vlnv xilinx.com:ip:axis_broadcaster:1.1 %s" % (cell_name))
                cmd.append("set_property -dict [list CONFIG.M_TDATA_NUM_BYTES.VALUE_SRC USER CONFIG.S_TDATA_NUM_BYTES.VALUE_SRC USER CONFIG.NUM_MI {%d}] [get_bd_cells %s]" % (num_mi, cell_name))
                cmd.append("set_property -dict [list CONFIG.M_TDATA_NUM_BYTES {%d} CONFIG.S_TDATA_NUM_BYTES {%d}] [get_bd_cells %s]" % (ibytes, ibytes, cell_name))
                cmd.append("connect_bd_net [get_bd_pins %s/aclk] [get_bd_pins %s/aclk]" % (hier_name, cell_name))
                cmd.append("connect_bd_net [get_bd_pins %s/aresetn] [get_bd_pins %s/aresetn]" % (hier_name, cell_name))
                if rep_hierarchical:
                    cmd.append("connect_bd_intf_net [get_bd_intf_pins %s/rep_root_bcast/M%02d_AXIS] [get_bd_intf_pins %s/S_AXIS]" % (hier_name, i, cell_name))
                else:
                    cmd.append("connect_bd_intf_net [get_bd_intf_pins %s/s_axis] [get_bd_intf_pins %s/S_AXIS]" % (hier_name, cell_name))
                for j in range(num_mi):
                    cmd.append("connect_bd_intf_net [get_bd_intf_pins %s/m_%d_axis] [get_bd_intf_pins %s/M%02d_AXIS]" % (hier_name, 16*i+j, cell_name, j))
            else:
                cmd.append("connect_bd_intf_net [get_bd_intf_pins %s/rep_root_bcast/M%02d_AXIS] [get_bd_intf_pins %s/m_%d_axis]" % (hier_name, i, hier_name, 16*i))

    return cmd


def axis_scatter(new_hier, nsplits, obits, parent_hier=None):
    obytes = int(math.ceil(obits/8))
    ibytes = int(math.ceil(obits*nsplits/8))
    assert nsplits<=256, "nsplits supported up to 256"
    cmd = []
    if parent_hier is not None:
        hier_name = parent_hier + "/" + new_hier
    else:
        hier_name = new_hier
    # Create hierarchy
    cmd.append("create_bd_cell -type hier %s" %(hier_name))

    # Create interface ports (AXI Stream in/out)
    cmd.append("create_bd_pin -dir I -type clk %s/aclk" % (hier_name))
    cmd.append("create_bd_pin -dir I -type rst %s/aresetn" % (hier_name))
    for i in range(nsplits):
        cmd.append("create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 %s/m_%d_axis" % (hier_name, i))
    cmd.append("create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 %s/s_axis" % (hier_name))

    # Instantiate Broadcaster(s) for nsplits
    if nsplits==1:
        cmd.append("connect_bd_intf_net [get_bd_intf_pins %s/s_axis] [get_bd_intf_pins %s/m_0_axis]" % (hier_name, hier_name))
    else:
        split_hierarchical = True if nsplits>16 else False
        split_nbcast = int(math.ceil(nsplits/16))
        #instantiate root broadcaster if nsplits > 16
        if split_hierarchical:
            root_bcast_obytes = int(math.ceil(16*obits/8))
            cell_name = "%s/split_root_bcast" % (hier_name)
            cmd.append("create_bd_cell -type ip -vlnv xilinx.com:ip:axis_broadcaster:1.1 %s" % (cell_name))
            cmd.append("set_property -dict [list CONFIG.M_TDATA_NUM_BYTES.VALUE_SRC USER CONFIG.S_TDATA_NUM_BYTES.VALUE_SRC USER CONFIG.NUM_MI {%d}] [get_bd_cells %s]" % (split_nbcast, cell_name))
            cmd.append("set_property -dict [list CONFIG.S_TDATA_NUM_BYTES {%d} CONFIG.M_TDATA_NUM_BYTES {%d}] [get_bd_cells %s]" % (ibytes, root_bcast_obytes, cell_name))
            cmd.append("connect_bd_net [get_bd_pins %s/aclk] [get_bd_pins %s/aclk]" % (hier_name, cell_name))
            cmd.append("connect_bd_net [get_bd_pins %s/aresetn] [get_bd_pins %s/aresetn]" % (hier_name, cell_name))
            cmd.append("connect_bd_intf_net [get_bd_intf_pins %s/s_axis] [get_bd_intf_pins %s/S_AXIS]" % (hier_name, cell_name))
            for i in range(split_nbcast):
                chunk_bits = obits*min(16, nsplits-i*16)
                pad_bits = 8*root_bcast_obytes - chunk_bits
                pad_string = "" if pad_bits == 0 else ("%d'b" %(pad_bits) + "0"*pad_bits + ",")
                cmd.append("set_property -dict [list CONFIG.M%02d_TDATA_REMAP {%stdata[%d:%d]}] [get_bd_cells %s]" % (i, pad_string, i*16*obits+chunk_bits-1, i*16*obits, cell_name))
        #instantiate leaf broadcaster(s)
        for i in range(split_nbcast):
            num_mi = min(16, nsplits - i*16)
            bcast_ibytes = root_bcast_obytes if split_hierarchical else ibytes
            cell_name = "%s/split_bcast_%d" % (hier_name, i)
            chunk_bits = obits
            pad_bits = 8*obytes - obits
            pad_string = "" if pad_bits == 0 else ("%d'b" %(pad_bits) + "0"*pad_bits + ",")
            if num_mi>1:
                cmd.append("create_bd_cell -type ip -vlnv xilinx.com:ip:axis_broadcaster:1.1 %s" % (cell_name))
                cmd.append("set_property -dict [list CONFIG.M_TDATA_NUM_BYTES.VALUE_SRC USER CONFIG.S_TDATA_NUM_BYTES.VALUE_SRC USER CONFIG.NUM_MI {%d}] [get_bd_cells %s]" % (num_mi, cell_name))
                cmd.append("set_property -dict [list CONFIG.S_TDATA_NUM_BYTES {%d} CONFIG.M_TDATA_NUM_BYTES {%d}] [get_bd_cells %s]" % (bcast_ibytes, obytes, cell_name))
                for j in range(num_mi):
                    cmd.append("set_property -dict [list CONFIG.M%02d_TDATA_REMAP {%stdata[%d:%d]}] [get_bd_cells %s]" % (j, pad_string, (j+1)*obits-1, j*obits, cell_name))
                if split_hierarchical:
                    cmd.append("connect_bd_intf_net [get_bd_intf_pins %s/split_root_bcast/M%02d_AXIS] [get_bd_intf_pins %s/S_AXIS]" % (hier_name, i, cell_name))
                else:
                    cmd.append("connect_bd_intf_net [get_bd_intf_pins %s/s_axis] [get_bd_intf_pins %s/S_AXIS]" % (hier_name, cell_name))
                for j in range(num_mi):
                    cmd.append("connect_bd_intf_net [get_bd_intf_pins %s/m_%d_axis] [get_bd_intf_pins %s/M%02d_AXIS]" % (hier_name, 16*i+j, cell_name, j))
            else:
                cmd.append("create_bd_cell -type ip -vlnv xilinx.com:ip:axis_subset_converter:1.1 %s" % (cell_name))
                cmd.append("set_property -dict [list CONFIG.M_TDATA_NUM_BYTES.VALUE_SRC USER CONFIG.S_TDATA_NUM_BYTES.VALUE_SRC USER] [get_bd_cells %s]" %(cell_name))
                cmd.append("set_property -dict [list CONFIG.S_TDATA_NUM_BYTES {%d} CONFIG.M_TDATA_NUM_BYTES {%d} CONFIG.TDATA_REMAP {%stdata[%d:0]}] [get_bd_cells %s]" %(bcast_ibytes, obytes, pad_string, obits-1, cell_name))
                cmd.append("connect_bd_intf_net [get_bd_intf_pins %s/S_AXIS] [get_bd_intf_pins %s/split_root_bcast/M%02d_AXIS]" % (cell_name, hier_name, i))       
                cmd.append("connect_bd_intf_net [get_bd_intf_pins %s/m_%d_axis] [get_bd_intf_pins %s/M_AXIS]" % (hier_name, 16*i, cell_name))
            cmd.append("connect_bd_net [get_bd_pins %s/aclk] [get_bd_pins %s/aclk]" % (hier_name, cell_name))
            cmd.append("connect_bd_net [get_bd_pins %s/aresetn] [get_bd_pins %s/aresetn]" % (hier_name, cell_name))


    return cmd

if __name__ == "__main__":
    import sys
    print("\n".join(axis_gather_bcast_scatter("foo", int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))))