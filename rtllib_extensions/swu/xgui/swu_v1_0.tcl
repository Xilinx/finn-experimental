# Definitional proc to organize widgets for parameters.
proc init_gui { IPINST } {
  ipgui::add_param $IPINST -name "Component_Name"
  #Adding Page
  set Page_0 [ipgui::add_page $IPINST -name "Page 0"]
  ipgui::add_param $IPINST -name "BUFFER_SIZE" -parent ${Page_0}
  ipgui::add_param $IPINST -name "IFMChannels" -parent ${Page_0}
  ipgui::add_param $IPINST -name "IFMHeight" -parent ${Page_0}
  ipgui::add_param $IPINST -name "IFMWidth" -parent ${Page_0}
  ipgui::add_param $IPINST -name "IP_PRECISION" -parent ${Page_0}
  ipgui::add_param $IPINST -name "KERNEL_HEIGHT" -parent ${Page_0}
  ipgui::add_param $IPINST -name "KERNEL_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "MMV_IN" -parent ${Page_0}
  ipgui::add_param $IPINST -name "MMV_OUT" -parent ${Page_0}
  ipgui::add_param $IPINST -name "OFMDIM_MOD_MMV" -parent ${Page_0}
  ipgui::add_param $IPINST -name "OFMHeight" -parent ${Page_0}
  ipgui::add_param $IPINST -name "OFMWidth" -parent ${Page_0}
  ipgui::add_param $IPINST -name "PADDING_HEIGHT" -parent ${Page_0}
  ipgui::add_param $IPINST -name "PADDING_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "RAM_STYLE" -parent ${Page_0}
  ipgui::add_param $IPINST -name "SIMD" -parent ${Page_0}
  ipgui::add_param $IPINST -name "STRIDE" -parent ${Page_0}


}

proc update_PARAM_VALUE.BUFFER_SIZE { PARAM_VALUE.BUFFER_SIZE } {
	# Procedure called to update BUFFER_SIZE when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.BUFFER_SIZE { PARAM_VALUE.BUFFER_SIZE } {
	# Procedure called to validate BUFFER_SIZE
	return true
}

proc update_PARAM_VALUE.IFMChannels { PARAM_VALUE.IFMChannels } {
	# Procedure called to update IFMChannels when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.IFMChannels { PARAM_VALUE.IFMChannels } {
	# Procedure called to validate IFMChannels
	return true
}

proc update_PARAM_VALUE.IFMHeight { PARAM_VALUE.IFMHeight } {
	# Procedure called to update IFMHeight when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.IFMHeight { PARAM_VALUE.IFMHeight } {
	# Procedure called to validate IFMHeight
	return true
}

proc update_PARAM_VALUE.IFMWidth { PARAM_VALUE.IFMWidth } {
	# Procedure called to update IFMWidth when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.IFMWidth { PARAM_VALUE.IFMWidth } {
	# Procedure called to validate IFMWidth
	return true
}

proc update_PARAM_VALUE.IP_PRECISION { PARAM_VALUE.IP_PRECISION } {
	# Procedure called to update IP_PRECISION when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.IP_PRECISION { PARAM_VALUE.IP_PRECISION } {
	# Procedure called to validate IP_PRECISION
	return true
}

proc update_PARAM_VALUE.KERNEL_HEIGHT { PARAM_VALUE.KERNEL_HEIGHT } {
	# Procedure called to update KERNEL_HEIGHT when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.KERNEL_HEIGHT { PARAM_VALUE.KERNEL_HEIGHT } {
	# Procedure called to validate KERNEL_HEIGHT
	return true
}

proc update_PARAM_VALUE.KERNEL_WIDTH { PARAM_VALUE.KERNEL_WIDTH } {
	# Procedure called to update KERNEL_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.KERNEL_WIDTH { PARAM_VALUE.KERNEL_WIDTH } {
	# Procedure called to validate KERNEL_WIDTH
	return true
}

proc update_PARAM_VALUE.MMV_IN { PARAM_VALUE.MMV_IN } {
	# Procedure called to update MMV_IN when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.MMV_IN { PARAM_VALUE.MMV_IN } {
	# Procedure called to validate MMV_IN
	return true
}

proc update_PARAM_VALUE.MMV_OUT { PARAM_VALUE.MMV_OUT } {
	# Procedure called to update MMV_OUT when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.MMV_OUT { PARAM_VALUE.MMV_OUT } {
	# Procedure called to validate MMV_OUT
	return true
}

proc update_PARAM_VALUE.OFMDIM_MOD_MMV { PARAM_VALUE.OFMDIM_MOD_MMV } {
	# Procedure called to update OFMDIM_MOD_MMV when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.OFMDIM_MOD_MMV { PARAM_VALUE.OFMDIM_MOD_MMV } {
	# Procedure called to validate OFMDIM_MOD_MMV
	return true
}

proc update_PARAM_VALUE.OFMHeight { PARAM_VALUE.OFMHeight } {
	# Procedure called to update OFMHeight when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.OFMHeight { PARAM_VALUE.OFMHeight } {
	# Procedure called to validate OFMHeight
	return true
}

proc update_PARAM_VALUE.OFMWidth { PARAM_VALUE.OFMWidth } {
	# Procedure called to update OFMWidth when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.OFMWidth { PARAM_VALUE.OFMWidth } {
	# Procedure called to validate OFMWidth
	return true
}

proc update_PARAM_VALUE.PADDING_HEIGHT { PARAM_VALUE.PADDING_HEIGHT } {
	# Procedure called to update PADDING_HEIGHT when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.PADDING_HEIGHT { PARAM_VALUE.PADDING_HEIGHT } {
	# Procedure called to validate PADDING_HEIGHT
	return true
}

proc update_PARAM_VALUE.PADDING_WIDTH { PARAM_VALUE.PADDING_WIDTH } {
	# Procedure called to update PADDING_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.PADDING_WIDTH { PARAM_VALUE.PADDING_WIDTH } {
	# Procedure called to validate PADDING_WIDTH
	return true
}

proc update_PARAM_VALUE.RAM_STYLE { PARAM_VALUE.RAM_STYLE } {
	# Procedure called to update RAM_STYLE when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.RAM_STYLE { PARAM_VALUE.RAM_STYLE } {
	# Procedure called to validate RAM_STYLE
	return true
}

proc update_PARAM_VALUE.SIMD { PARAM_VALUE.SIMD } {
	# Procedure called to update SIMD when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SIMD { PARAM_VALUE.SIMD } {
	# Procedure called to validate SIMD
	return true
}

proc update_PARAM_VALUE.STRIDE { PARAM_VALUE.STRIDE } {
	# Procedure called to update STRIDE when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.STRIDE { PARAM_VALUE.STRIDE } {
	# Procedure called to validate STRIDE
	return true
}


proc update_MODELPARAM_VALUE.SIMD { MODELPARAM_VALUE.SIMD PARAM_VALUE.SIMD } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.SIMD}] ${MODELPARAM_VALUE.SIMD}
}

proc update_MODELPARAM_VALUE.STRIDE { MODELPARAM_VALUE.STRIDE PARAM_VALUE.STRIDE } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.STRIDE}] ${MODELPARAM_VALUE.STRIDE}
}

proc update_MODELPARAM_VALUE.IFMChannels { MODELPARAM_VALUE.IFMChannels PARAM_VALUE.IFMChannels } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.IFMChannels}] ${MODELPARAM_VALUE.IFMChannels}
}

proc update_MODELPARAM_VALUE.KERNEL_HEIGHT { MODELPARAM_VALUE.KERNEL_HEIGHT PARAM_VALUE.KERNEL_HEIGHT } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.KERNEL_HEIGHT}] ${MODELPARAM_VALUE.KERNEL_HEIGHT}
}

proc update_MODELPARAM_VALUE.KERNEL_WIDTH { MODELPARAM_VALUE.KERNEL_WIDTH PARAM_VALUE.KERNEL_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.KERNEL_WIDTH}] ${MODELPARAM_VALUE.KERNEL_WIDTH}
}

proc update_MODELPARAM_VALUE.RAM_STYLE { MODELPARAM_VALUE.RAM_STYLE PARAM_VALUE.RAM_STYLE } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.RAM_STYLE}] ${MODELPARAM_VALUE.RAM_STYLE}
}

proc update_MODELPARAM_VALUE.IFMWidth { MODELPARAM_VALUE.IFMWidth PARAM_VALUE.IFMWidth } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.IFMWidth}] ${MODELPARAM_VALUE.IFMWidth}
}

proc update_MODELPARAM_VALUE.IFMHeight { MODELPARAM_VALUE.IFMHeight PARAM_VALUE.IFMHeight } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.IFMHeight}] ${MODELPARAM_VALUE.IFMHeight}
}

proc update_MODELPARAM_VALUE.PADDING_WIDTH { MODELPARAM_VALUE.PADDING_WIDTH PARAM_VALUE.PADDING_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.PADDING_WIDTH}] ${MODELPARAM_VALUE.PADDING_WIDTH}
}

proc update_MODELPARAM_VALUE.PADDING_HEIGHT { MODELPARAM_VALUE.PADDING_HEIGHT PARAM_VALUE.PADDING_HEIGHT } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.PADDING_HEIGHT}] ${MODELPARAM_VALUE.PADDING_HEIGHT}
}

proc update_MODELPARAM_VALUE.OFMWidth { MODELPARAM_VALUE.OFMWidth PARAM_VALUE.OFMWidth } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.OFMWidth}] ${MODELPARAM_VALUE.OFMWidth}
}

proc update_MODELPARAM_VALUE.OFMHeight { MODELPARAM_VALUE.OFMHeight PARAM_VALUE.OFMHeight } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.OFMHeight}] ${MODELPARAM_VALUE.OFMHeight}
}

proc update_MODELPARAM_VALUE.IP_PRECISION { MODELPARAM_VALUE.IP_PRECISION PARAM_VALUE.IP_PRECISION } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.IP_PRECISION}] ${MODELPARAM_VALUE.IP_PRECISION}
}

proc update_MODELPARAM_VALUE.MMV_IN { MODELPARAM_VALUE.MMV_IN PARAM_VALUE.MMV_IN } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.MMV_IN}] ${MODELPARAM_VALUE.MMV_IN}
}

proc update_MODELPARAM_VALUE.MMV_OUT { MODELPARAM_VALUE.MMV_OUT PARAM_VALUE.MMV_OUT } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.MMV_OUT}] ${MODELPARAM_VALUE.MMV_OUT}
}

proc update_MODELPARAM_VALUE.BUFFER_SIZE { MODELPARAM_VALUE.BUFFER_SIZE PARAM_VALUE.BUFFER_SIZE } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.BUFFER_SIZE}] ${MODELPARAM_VALUE.BUFFER_SIZE}
}

proc update_MODELPARAM_VALUE.OFMDIM_MOD_MMV { MODELPARAM_VALUE.OFMDIM_MOD_MMV PARAM_VALUE.OFMDIM_MOD_MMV } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.OFMDIM_MOD_MMV}] ${MODELPARAM_VALUE.OFMDIM_MOD_MMV}
}

