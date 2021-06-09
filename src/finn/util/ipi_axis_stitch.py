import math

# create the TCL to instantiate a broadcast-then-scatter tree in a new BD hierarchy
def axis_gather_bcast_scatter(new_hier, njoins, nreplicas, nsplits, ibytes, parent_hier=None):
    nbytes = ibytes*njoins
    assert njoins<=256, "njoins supported up to 256"
    assert nreplicas<=256, "nreplicas supported up to 256"
    assert nsplits<=256, "nsplits supported up to 256"
    assert nbytes % nsplits == 0, "nsplits must divide nbytes"
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
    if njoins > 1:
        join_hierarchical = True if njoins>16 else False
        join_ncomb = int(math.ceil(njoins/16))
        #instantiate leaf combiner(s)
        for i in range(join_ncomb):
            num_si = min(16, njoins - i*16)
            cell_name = "%s/join_comb_%d" % (hier_name, i)
            cmd.append("create_bd_cell -type ip -vlnv xilinx.com:ip:axis_combiner:1.1 %s" % (cell_name))
            cmd.append("set_property -dict [list CONFIG.TDATA_NUM_BYTES.VALUE_SRC USER CONFIG.TDATA_NUM_BYTES {%d} CONFIG.NUM_SI {%d}] [get_bd_cells %s]" % (ibytes, num_si, cell_name))
            cmd.append("connect_bd_net [get_bd_pins %s/aclk] [get_bd_pins %s/aclk]" % (hier_name, cell_name))
            cmd.append("connect_bd_net [get_bd_pins %s/aresetn] [get_bd_pins %s/aresetn]" % (hier_name, cell_name))
            for j in range(num_si):
                cmd.append("connect_bd_intf_net [get_bd_intf_pins %s/s_%d_axis] [get_bd_intf_pins %s/S%02d_AXIS]" % (hier_name, 16*i+j, cell_name, j))       
        if join_hierarchical:
            cell_name = "%s/join_root_comb" % (hier_name)
            cmd.append("create_bd_cell -type ip -vlnv xilinx.com:ip:axis_combiner:1.1 %s" % (cell_name))
            cmd.append("set_property -dict [list CONFIG.TDATA_NUM_BYTES.VALUE_SRC USER CONFIG.TDATA_NUM_BYTES {%d} CONFIG.NUM_SI {%d}] [get_bd_cells %s]" % (16*ibytes, join_ncomb, cell_name))
            cmd.append("connect_bd_net [get_bd_pins %s/aclk] [get_bd_pins %s/aclk]" % (hier_name, cell_name))
            cmd.append("connect_bd_net [get_bd_pins %s/aresetn] [get_bd_pins %s/aresetn]" % (hier_name, cell_name))
            for i in range(join_ncomb):
                cmd.append("connect_bd_intf_net [get_bd_intf_pins %s/join_comb_%d/M_AXIS] [get_bd_intf_pins %s/S%02d_AXIS]" % (hier_name, i, cell_name, i))       
            # if the inputs to the root combiner are asymmetrical (because of asymmetrical leaf combiners)
            # then chop off the excess bytes with a subset converter
            if njoins % 16 != 0:
                #TODO:
                cell_name = "%s/join_root_conv" % (hier_name)
                cmd.append("create_bd_cell -type ip -vlnv xilinx.com:ip:axis_subset_converter:1.1 %s" % (cell_name))
                cmd.append("set_property -dict [list CONFIG.M_TDATA_NUM_BYTES.VALUE_SRC USER CONFIG.S_TDATA_NUM_BYTES.VALUE_SRC USER] [get_bd_cells %s]" %(cell_name))
                cmd.append("set_property -dict [list CONFIG.S_TDATA_NUM_BYTES {%d} CONFIG.M_TDATA_NUM_BYTES {%d} CONFIG.TDATA_REMAP {tdata[%d:0]}] [get_bd_cells %s]" %(16*join_ncomb, ibytes*njoins, 8*ibytes*njoins-1, cell_name))
                cmd.append("connect_bd_net [get_bd_pins %s/aclk] [get_bd_pins %s/aclk]" % (hier_name, cell_name))
                cmd.append("connect_bd_net [get_bd_pins %s/aresetn] [get_bd_pins %s/aresetn]" % (hier_name, cell_name))
                cmd.append("connect_bd_intf_net [get_bd_intf_pins %s/S_AXIS] [get_bd_intf_pins %s/join_root_comb/M_AXIS]" % (cell_name, hier_name))       
                join_output = "%s/join_root_conv/M_AXIS" % (hier_name)
            else:
                join_output = "%s/join_root_comb/M_AXIS" % (hier_name)
        else:
            join_output = "%s/join_comb_0/M_AXIS" % (hier_name)
    else:
        join_output = "%s/s_0_axis" % (hier_name)

    if nreplicas==1 and nsplits==1:
        cmd.append("connect_bd_intf_net [get_bd_intf_pins %s] [get_bd_intf_pins %s/m_0_0_axis]" % (join_output, hier_name))
        return cmd

    # Instantiate Broadcaster(s) for nreplicas
    if nreplicas>1:
        rep_hierarchical = True if nreplicas>16 else False
        rep_nbcast = int(math.ceil(nreplicas/16))
        #instantiate root broadcaster if nreplicas > 16
        if rep_hierarchical:
            cell_name = "%s/rep_root_bcast" % (hier_name)
            cmd.append("create_bd_cell -type ip -vlnv xilinx.com:ip:axis_broadcaster:1.1 %s" % (cell_name))
            cmd.append("set_property -dict [list CONFIG.M_TDATA_NUM_BYTES.VALUE_SRC USER CONFIG.S_TDATA_NUM_BYTES.VALUE_SRC USER CONFIG.NUM_MI {%d}] [get_bd_cells %s]" % (rep_nbcast, cell_name))
            cmd.append("set_property -dict [list CONFIG.M_TDATA_NUM_BYTES {%d} CONFIG.S_TDATA_NUM_BYTES {%d}] [get_bd_cells %s]" % (nbytes, nbytes, cell_name))
            cmd.append("connect_bd_net [get_bd_pins %s/aclk] [get_bd_pins %s/aclk]" % (hier_name, cell_name))
            cmd.append("connect_bd_net [get_bd_pins %s/aresetn] [get_bd_pins %s/aresetn]" % (hier_name, cell_name))
        #instantiate leaf broadcaster(s)
        for i in range(rep_nbcast):
            num_mi = min(16, nreplicas - i*16)
            cell_name = "%s/rep_bcast_%d" % (hier_name, i)
            cmd.append("create_bd_cell -type ip -vlnv xilinx.com:ip:axis_broadcaster:1.1 %s" % (cell_name))
            cmd.append("set_property -dict [list CONFIG.M_TDATA_NUM_BYTES.VALUE_SRC USER CONFIG.S_TDATA_NUM_BYTES.VALUE_SRC USER CONFIG.NUM_MI {%d}] [get_bd_cells %s]" % (num_mi, cell_name))
            cmd.append("set_property -dict [list CONFIG.M_TDATA_NUM_BYTES {%d} CONFIG.S_TDATA_NUM_BYTES {%d}] [get_bd_cells %s]" % (nbytes, nbytes, cell_name))
            cmd.append("connect_bd_net [get_bd_pins %s/aclk] [get_bd_pins %s/aclk]" % (hier_name, cell_name))
            cmd.append("connect_bd_net [get_bd_pins %s/aresetn] [get_bd_pins %s/aresetn]" % (hier_name, cell_name))
        #make connections
        if rep_hierarchical:
            cmd.append("connect_bd_intf_net [get_bd_intf_pins %s] [get_bd_intf_pins %s/rep_root_bcast/S_AXIS]" % (join_output, hier_name))
            for i in range(rep_nbcast):
                cmd.append("connect_bd_intf_net [get_bd_intf_pins %s/rep_root_bcast/M%02d_AXIS] [get_bd_intf_pins %s/rep_bcast_%d/S_AXIS]" % (hier_name, i, hier_name, i))
        else:
            cmd.append("connect_bd_intf_net [get_bd_intf_pins %s] [get_bd_intf_pins %s/rep_bcast_0/S_AXIS]" % (join_output, hier_name))

    # Instantiate Splitter(s) for nsplits
    split_hierarchical = True if nsplits>16 else False
    split_nscatter = int(math.ceil(nsplits/16))
    split_nbytes = nbytes//nsplits
    for m in range(nreplicas):
        if nsplits > 1:
            #instantiate root splitter if nsplits > 16
            if split_hierarchical:
                cell_name = "%s/split_root_scatter_%d" % (hier_name, m)
                cmd.append("create_bd_cell -type ip -vlnv xilinx.com:ip:axis_broadcaster:1.1 %s" % (cell_name))
                cmd.append("set_property -dict [list CONFIG.M_TDATA_NUM_BYTES.VALUE_SRC USER CONFIG.S_TDATA_NUM_BYTES.VALUE_SRC USER CONFIG.NUM_MI {%d}] [get_bd_cells %s]" % (split_nscatter, cell_name))
                cmd.append("set_property -dict [list CONFIG.M_TDATA_NUM_BYTES {%d} CONFIG.S_TDATA_NUM_BYTES {%d}] [get_bd_cells %s]" % (nbytes, nbytes, cell_name))
                cmd.append("connect_bd_net [get_bd_pins %s/aclk] [get_bd_pins %s/aclk]" % (hier_name, cell_name))
                cmd.append("connect_bd_net [get_bd_pins %s/aresetn] [get_bd_pins %s/aresetn]" % (hier_name, cell_name))
            #instantiate leaf splitter(s)
            for i in range(split_nscatter):
                num_mi = min(16, nsplits - i*16)
                cell_name = "%s/split_scatter_%d_%d" % (hier_name, m, i)
                cmd.append("create_bd_cell -type ip -vlnv xilinx.com:ip:axis_broadcaster:1.1 %s" % (cell_name))
                cmd.append("set_property -dict [list CONFIG.M_TDATA_NUM_BYTES.VALUE_SRC USER CONFIG.S_TDATA_NUM_BYTES.VALUE_SRC USER CONFIG.NUM_MI {%d}] [get_bd_cells %s]" % (num_mi, cell_name))
                cmd.append("set_property -dict [list CONFIG.M_TDATA_NUM_BYTES {%d} CONFIG.S_TDATA_NUM_BYTES {%d}] [get_bd_cells %s]" % (split_nbytes, nbytes, cell_name))
                cmd.append("connect_bd_net [get_bd_pins %s/aclk] [get_bd_pins %s/aclk]" % (hier_name, cell_name))
                cmd.append("connect_bd_net [get_bd_pins %s/aresetn] [get_bd_pins %s/aresetn]" % (hier_name, cell_name))
                for j in range(num_mi):
                    cmd.append("set_property -dict [list CONFIG.M%02d_TDATA_REMAP {tdata[%d:%d]}] [get_bd_cells %s]" % (j, (j+1)*8*split_nbytes-1, j*8*split_nbytes, cell_name))
            #make connections
            if split_hierarchical:
                if nreplicas == 1:
                    cmd.append("connect_bd_intf_net [get_bd_intf_pins %s] [get_bd_intf_pins %s/split_root_scatter_0/S_AXIS]" % (join_output, hier_name))
                else:
                    cmd.append("connect_bd_intf_net [get_bd_intf_pins %s/rep_bcast_%d/M%02d_AXIS] [get_bd_intf_pins %s/split_root_scatter_%d/S_AXIS]" % (hier_name, m//16, m%16, hier_name, m))
                for i in range(split_nscatter):
                    cmd.append("connect_bd_intf_net [get_bd_intf_pins %s/split_root_scatter_%d/M%02d_AXIS] [get_bd_intf_pins %s/split_scatter_%d_%d/S_AXIS]" % (hier_name, m, i, hier_name, m, i))
            else:
                if nreplicas == 1:
                    cmd.append("connect_bd_intf_net [get_bd_intf_pins %s] [get_bd_intf_pins %s/split_scatter_0_0/S_AXIS]" % (join_output, hier_name))
                else:
                    cmd.append("connect_bd_intf_net [get_bd_intf_pins %s/rep_bcast_%d/M%02d_AXIS] [get_bd_intf_pins %s/split_scatter_%d_0/S_AXIS]" % (hier_name, m//16, m%16, hier_name, m))
            # connect outputs
            for i in range(split_nscatter):
                for j in range(min(16, nsplits - i*16)):
                    cmd.append("connect_bd_intf_net [get_bd_intf_pins %s/m_%d_%d_axis] [get_bd_intf_pins %s/split_scatter_%d_%d/M%02d_AXIS]" % (hier_name, m, i*16+j, hier_name, m, i, j))
        else:
            cmd.append("connect_bd_intf_net [get_bd_intf_pins %s/m_%d_0_axis] [get_bd_intf_pins %s/rep_bcast_%d/M%02d_AXIS]" % (hier_name, m, hier_name, m//16, m%16))

    return cmd