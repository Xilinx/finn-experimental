# Definitional proc to organize widgets for parameters.
proc init_gui { IPINST } {
  ipgui::add_param $IPINST -name "Component_Name"
  #Adding Page
  set Page_0 [ipgui::add_page $IPINST -name "Page 0"]
  ipgui::add_param $IPINST -name "DEPTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "NFOLDS" -parent ${Page_0}
  ipgui::add_param $IPINST -name "RAM_STYLE" -parent ${Page_0}
  ipgui::add_param $IPINST -name "WIDTH" -parent ${Page_0}


}

proc update_PARAM_VALUE.DEPTH { PARAM_VALUE.DEPTH } {
	# Procedure called to update DEPTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.DEPTH { PARAM_VALUE.DEPTH } {
	# Procedure called to validate DEPTH
	return true
}

proc update_PARAM_VALUE.NFOLDS { PARAM_VALUE.NFOLDS } {
	# Procedure called to update NFOLDS when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.NFOLDS { PARAM_VALUE.NFOLDS } {
	# Procedure called to validate NFOLDS
	return true
}

proc update_PARAM_VALUE.RAM_STYLE { PARAM_VALUE.RAM_STYLE } {
	# Procedure called to update RAM_STYLE when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.RAM_STYLE { PARAM_VALUE.RAM_STYLE } {
	# Procedure called to validate RAM_STYLE
	return true
}

proc update_PARAM_VALUE.WIDTH { PARAM_VALUE.WIDTH } {
	# Procedure called to update WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.WIDTH { PARAM_VALUE.WIDTH } {
	# Procedure called to validate WIDTH
	return true
}


proc update_MODELPARAM_VALUE.WIDTH { MODELPARAM_VALUE.WIDTH PARAM_VALUE.WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.WIDTH}] ${MODELPARAM_VALUE.WIDTH}
}

proc update_MODELPARAM_VALUE.DEPTH { MODELPARAM_VALUE.DEPTH PARAM_VALUE.DEPTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DEPTH}] ${MODELPARAM_VALUE.DEPTH}
}

proc update_MODELPARAM_VALUE.NFOLDS { MODELPARAM_VALUE.NFOLDS PARAM_VALUE.NFOLDS } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.NFOLDS}] ${MODELPARAM_VALUE.NFOLDS}
}

proc update_MODELPARAM_VALUE.RAM_STYLE { MODELPARAM_VALUE.RAM_STYLE PARAM_VALUE.RAM_STYLE } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.RAM_STYLE}] ${MODELPARAM_VALUE.RAM_STYLE}
}

