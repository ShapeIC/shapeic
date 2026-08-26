drc off
gds read /home/daniel/shapeic/shapeic-layout/lut_generation/work-cellkit-magic-smoke/simplediffpair_l0_w0_n0/layout.gds
load simplediffpair_l0p400_wf0p150_nf1
select top cell
flatten -dotoplabels simplediffpair_l0p400_wf0p150_nf1_flat
load simplediffpair_l0p400_wf0p150_nf1_flat
select top cell
extract do local
extract all
ext2spice lvs
ext2spice cthresh 0
ext2spice rthresh 0
ext2spice extresist on
ext2spice -o /home/daniel/shapeic/shapeic-layout/lut_generation/work-cellkit-magic-smoke/simplediffpair_l0_w0_n0/simplediffpair_l0p400_wf0p150_nf1.pex.spice
quit -noprompt
