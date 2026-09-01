* tb.inc

VDD vdd 0 1.8

VINP inp 0 DC 0.9 AC 0.5
VINN inn 0 DC 0.9 AC -0.5

XOTA inp inn out vdd 0 ota

CLOAD out 0 1p
