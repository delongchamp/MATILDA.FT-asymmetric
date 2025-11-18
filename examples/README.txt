This sub-directory has the input files to perform token simulations for the paper 
mentioned in the top-level.

The symmetric and asymmetric subdirectories have the files to model two types of 
calculations: one where the activity of polymer B is systematically increased 
(./symmetric/zb-up/) and one where it is decrease (./symmetric/zb-down/). These
calculations begin by performing a calculation in the grand-canonical ensemble
to generate an A-rich (zb-up) or B-rich (zb-down) state, then the activity is 
changed until the composition undergoes a step change. 

The symmetric folder simulates symmetric wetting nanoparticles (chiAP=chiBP=0),
and the asymmetric folder simulates asymmetric wetting. See the supporting
information from the paper for complete details.
