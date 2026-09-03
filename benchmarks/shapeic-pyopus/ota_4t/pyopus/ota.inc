
.subckt ota inp inn out vdd vss

X1 n1  inp tail tail sg13_lv_nmos w={w_n} l={l_n} ng={floor(w_n/10u)+1}
X2 out inn tail tail sg13_lv_nmos w={w_n} l={l_n} ng={floor(w_n/10u)+1}

X3 n1  n1  vdd vdd sg13_lv_pmos w={w_p} l={l_p} ng={floor(w_p/10u)+1}
X4 out n1  vdd vdd sg13_lv_pmos w={w_p} l={l_p} ng={floor(w_p/10u)+1}

Ibias tail vss 20u

.ends ota
