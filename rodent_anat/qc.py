"""
rodent_anat: QC data generation
"""
import os
import logging

import fsl.wrappers as fsl

from .utils import working_dir, makedirs

LOG = logging.getLogger(__name__)

def run(options):
    LOG.info("START: GENERATING QC DATA")

    outdir = os.path.abspath(options.output)
   
    with working_dir(outdir):
        makedirs("qc", exist_ok=True)

        templ_brain = os.path.join(options.template, "SIGMA_ExVivo_Brain_Template_Masked_No_OlfBulb.nii.gz")

        # Check that the registration was successful
        fsl.slicer("T2_templ_linear", templ_brain, i="0 1", a="qc/reg_linear.png")
        if options.nonlinreg:
            fsl.slicer("T2_templ", templ_brain, a="qc/reg_nonlin.png", i="0 1")

        # Check that the bias correction was successful, keep intensity range constant
        if not options.nobias:
            fsl.slicer("T2_brain", i="0 1", a="qc/bias_corr.png")
            fsl.slicer("T2_brain_nobiascorr", i="0 1", a="qc/no_bias_corr.png")
            fsl.slicer("T2_templ_linear_nobiascorr", templ_brain, a="qc/reg_linear_biased.png", i="0 1")

        # Check that the brain masking was successful in native and standard space
        fsl.slicer("T2_reorient", "T2_brain_mask_linear", a="qc/skull_strip_linear.png")
        fsl.slicer("T2_brain_templ_linear", templ_brain, out="qc/skull_strip_std_space_linear.png")
        if options.nonlinreg:
            fsl.slicer("T2_reorient", "T2_brain_mask", out="qc/skull_strip_nonlin.png")
            fsl.slicer("T2_brain_templ", templ_brain, out="qc/skull_strip_std_space_nonlin.png")

    LOG.info("DONE: GENERATING QC DATA")