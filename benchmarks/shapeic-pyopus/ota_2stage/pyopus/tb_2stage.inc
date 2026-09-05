* tb_2stage.inc

VDD vdd 0 1.5

*VINP inp 0 DC 0.9 
VINN inn 0 DC 0.9 AC 1

R1 out vfb 100000000
R2 vfb 0 300000000
C1 vfb 0 10

XOTA_2STAGE vfb inn out vdd 0 ota_2stage

*CLOAD out 0 1p
