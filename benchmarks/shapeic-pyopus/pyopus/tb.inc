* tb.inc

* Alimentacion nominal para los dispositivos core LV de SG13G2.
VDD vdd 0 1.2

VINP inp 0 DC 0.6 AC 0.5
VINN inn 0 DC 0.6 AC -0.5

XOTA inp inn out vdd 0 ota

CLOAD out 0 1p
