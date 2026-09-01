* OTA diferencial sencilla con dispositivos LV de IHP SG13G2.
* Los MOS del PDK son subcircuitos; por eso las instancias empiezan por X.

.subckt ota inp inn out vdd vss

X1 n1  inp tail vss sg13_lv_nmos w={w_n} l={l_n} ng=1
X2 out inn tail vss sg13_lv_nmos w={w_n} l={l_n} ng=1

X3 n1  n1  vdd vdd sg13_lv_pmos w={w_p} l={l_p} ng=1
X4 out n1  vdd vdd sg13_lv_pmos w={w_p} l={l_p} ng=1

Ibias tail vss 20u

.ends ota
