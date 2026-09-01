* OTA diferencial sencilla con carga activa
* Modelos de nivel 1: adecuados para comprobar el flujo PyOPUS/ngspice,
* no para predecir el comportamiento de una tecnologia real.
.model nch nmos level=1 vto=0.45  kp=200u lambda=0.02
.model pch pmos level=1 vto=-0.45 kp=80u  lambda=0.02

.subckt ota inp inn out vdd vss

M1 n1  inp tail vss nch W={w_n} L={l_n}
M2 out inn tail vss nch W={w_n} L={l_n}

M3 n1  n1  vdd vdd pch W={w_p} L={l_p}
M4 out n1  vdd vdd pch W={w_p} L={l_p}

Ibias tail vss 50u

.ends ota
