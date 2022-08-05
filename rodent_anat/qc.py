"""
rodent_anat: QC data generation
"""
import os
import logging
import argparse
import sys
import json
from pathlib import Path

import fsl.wrappers as fsl
from fsl.data.image import Image

from .utils import working_dir, makedirs, setup_logging, orient_cost

import numpy as np
import nibabel as nib

LOG = logging.getLogger(__name__)

def cnr(img_fname, mask_fname, r1_fname, r2_fname, threshold=0.5):
    img = Image(img_fname).data
    mask = Image(mask_fname).data > 0.5
    r1 = Image(r1_fname).data > threshold
    r2 = Image(r2_fname).data > threshold
    sig1 = np.mean(img[r1])
    sig2 = np.mean(img[r2])
    noise = np.std(img[~mask])
    return 0.655 * np.abs(sig1-sig2) / noise

def snr(img_fname, mask_fname):
    img = Image(img_fname).data
    mask = Image(mask_fname).data > 0.5

    sig = np.mean(img[mask])
    noise = np.std(img[~mask])
    return 0.655 * np.abs(sig) / noise

def sliceimg(img_fname, mask_fname=None, outname=None):
    if not outname:
        outname = os.path.basename(img_fname)
    img = Image(img_fname).data
    if mask_fname:
        mask = Image(mask_fname).data > 0.5
        img = img[mask]
    imax = int(np.percentile(img, 95))
    fsl.slicer(img_fname, mask_fname, i="0 %i" % imax, a=os.path.join("qc", outname + ".png"))

def run(options):
    LOG.info("START: GENERATING QC DATA")

    outdir = os.path.abspath(options.output)
   
    with working_dir(outdir):
        makedirs("qc", exist_ok=True)

        templ_brain = os.path.join(options.template, "SIGMA_ExVivo_Brain_Template_Masked_No_OlfBulb.nii.gz")

        # Check that the registration was successful
        sliceimg("T2_templ_linear", templ_brain, "reg_linear")
        if options.nonlinreg:
            sliceimg("T2_templ", templ_brain, "reg_nonlin")

        # Check that the bias correction was successful, keep intensity range constant
        if not options.nobias:
            sliceimg("T2_brain_linear", outname="biascorr")
            sliceimg("T2_brain_linear_nobiascorr", outname="nobiascorr")
            sliceimg("T2_templ_linear_nobiascorr", templ_brain, outname="reg_linear_nobiascorr")

        # Check that the brain masking was successful in native and standard space
        sliceimg("T2_reorient", "T2_brain_mask_linear", outname="skull_strip_linear")
        sliceimg("T2_brain_templ_linear", templ_brain, outname="skull_strip_std_space_linear")
        if options.nonlinreg:
            sliceimg("T2_reorient", "T2_brain_mask", outname="skull_strip_nonlin")
            sliceimg("T2_brain_templ", templ_brain, outname="skull_strip_std_space_nonlin")

        qcstats = {}

        # Calculate CNR
        qcstats["qc_gm_wm_cnr"] = cnr("T2_reorient", "T2_brain_mask", "seg/atlas/T2_brain_GM", "seg/atlas/T2_brain_WM")
        qcstats["qc_t2_snr"] = snr("T2_reorient", "T2_brain_mask")

        # Compare alignments
        qcstats["qc_linear_align_cost"] = orient_cost("T2_brain_linear", templ_brain, allow_translation=False)
        qcstats["qc_nonlin_align_cost"] = orient_cost("T2_brain", templ_brain, allow_translation=False)
        with open("qc/qc.json", "w") as f:
            json.dump(qcstats, f)

    LOG.info("DONE: GENERATING QC DATA")


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, **kwargs):
        argparse.ArgumentParser.__init__(self, prog="rodent_anat_qc", add_help=True, **kwargs)
        self.add_argument("output", help="Directory containing output of rodent_anat")
        self.add_argument("--debug", action="store_true", default=False, help="Enable debug output")
        self.add_argument("--nobias", help="Turn off steps that do bias field correction", action="store_true", default=False)
        self.add_argument("--nonlinreg", help="Non-linear registration method", choices=["ants", "mmorf"], default="mmorf")
        self.add_argument("--template", help="Path to directory containing SIGMA template files (default=$HOME/SIGMA_template)")

def main():
    parser = ArgumentParser()
    options = parser.parse_args()
    try:
        setup_logging(level="DEBUG" if options.debug else "WARN", log_stream=sys.stdout)
        options.input = os.path.basename(options.output)
        if not options.template:
            options.template = os.path.join(Path.home(), "SIGMA_template")
        run(options)
    except Exception as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        if options.debug:
            raise
        else:
            sys.exit(1)
