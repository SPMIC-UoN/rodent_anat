rodent_anat - Anatomical pipeline for rodent MRI based on fsl_anat interface
============================================================================

rodent_anat is an open source image pre-processing pipeline for rodent structural
MRI (i.e., T2-weighted scans). It is modelled on the fsl_anat interface. 

![rodent_anat](https://github.com/SPMIC-UoN/rodent_anat/assets/60778124/a8d55135-77e9-42f9-8769-0814b3849a0d)

What you can do with rodent_anat

The primary purpose of rodent_anat is to provide rodent structural MRI research 
with a standard, flexible, and robust image processing platform, which can 
generalise across scanners and acquisition protocols. 

rodent_anat is used in conjunction with https://github.com/SPMIC-UoN/brkraw 
and https://github.com/jennahanmer/DiffPreproc_rodent. 
https://github.com/SPMIC-UoN/brkraw is a verion of brkraw that has been 
modified such that it is better equipped for diffusion weighted imaging. 
https://github.com/jennahanmer/DiffPreproc_rodent preprocesses rodent dw-scans. 
It requires, as input, the T2-weighted scan and brain mask, created by
rodent_anat. 

Preprocessing


Dependencies

This pipeline depends on the conversion of Paravision data using a modified verion of brkraw 
(https://github.com/SPMIC-UoN/brkraw). 
