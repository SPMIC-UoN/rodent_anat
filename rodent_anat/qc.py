"""
rodent_anat: QC data generation
"""
import os
import logging

import fsl.wrappers as fsl

from .utils import working_dir, makedirs

LOG = logging.getLogger(__name__)

def run(options):
    LOG.info("GENERATING QC DATA")

    outdir = os.path.abspath(options.output)
   
    with working_dir(outdir):
        makedirs("qc", exist_ok=True))

        # Check that the linear registration of the T2 brain to the standard template was successful
        templ_brain = os.path.join(options.template, "SIGMA_ExVivo_Brain_Template_Masked_No_OlfBulb.nii.gz")
        fsl.slicer("tmp/T2_brain_to_templ", templ_brain, a="qc/brain_reorient.png")

        # Check that the brain masking was successful
        fsl.slicer("T2_reorient", "T2_mask_dil", a="qc/skull_strip.png")

        # Check that the bias correction was successful, keep intensity range constant
        fsl.slicer("fast/T2_brain_ud", i="0 1", a="qc/bias_corr.png")
        fsl.slicer("tmp/T2_brain", i="0 1", a="qc/no_bias_corr.png")

        # Check that the refined linear registration was successful
        fsl.slicer("T2_to_templ_linear", templ_brain, i="0 1", a="qc/linear.png")
        fsl.slicer("T2_to_templ_linear_biased", templ_brain, a="qc/linear_biased.png", i="0 1")

        if options.nonlinreg == "ants":
            # Check that this is working correctly
            fsl.slicer("ANTs/T2_to_templ_ANTs_Warped", templ_brain, a="qc/ANTS.png")

        elif options.nonlinreg == "mmorf":
            # Check that this is working correctly
            fsl.slicer("mmorf/T2_to_templ_mmorf", templ_brain, out="qc/mmorf.png")

        # Check that the brain masking was successful
        fsl.slicer("T2_reorient_ud", "T2_mask", out="qc/skull_strip.png")

        # Check that this is working correctly
        fsl.slicer("T2_brain_ud_to_templ", templ_brain, out="qc/skull_strip_std_space.png")
