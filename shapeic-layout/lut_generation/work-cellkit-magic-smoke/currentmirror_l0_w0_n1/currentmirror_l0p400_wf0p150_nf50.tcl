drc off
gds read /home/daniel/shapeic/shapeic-layout/lut_generation/work-cellkit-magic-smoke/currentmirror_l0_w0_n1/layout.gds
load currentmirror_l0p400_wf0p150_nf50
select top cell
flatten -dotoplabels currentmirror_l0p400_wf0p150_nf50_flat
load currentmirror_l0p400_wf0p150_nf50_flat
select top cell
extract do local
extract all
ext2spice lvs
ext2spice cthresh 0
ext2spice rthresh 0
ext2spice extresist on
ext2spice -o /home/daniel/shapeic/shapeic-layout/lut_generation/work-cellkit-magic-smoke/currentmirror_l0_w0_n1/currentmirror_l0p400_wf0p150_nf50.pex.spice
quit -noprompt
