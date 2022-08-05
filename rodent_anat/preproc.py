"""
rodent_anat: Command line tool to preprocess raw Bruker data
"""
import argparse
import os
import logging
import sys
import traceback

import numpy as np

from .utils import makedirs, setup_logging, sidecar
from .nifti_convert import convert_to_nifti
from .categorize import categorize_niftis, ANAT_BEST, DTI
from .dti import fix_dti_orientation, standardize_dti
from .reorient import reorient_niftis

LOG = logging.getLogger(__name__)

class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, **kwargs):
        argparse.ArgumentParser.__init__(self, prog="rodent_preproc", add_help=True, **kwargs)

        group = self.add_argument_group("Input/output Options")
        group.add_argument("-i", "--input", required=True, help="Directory containing raw Bruker data")
        group.add_argument("-o", "--output", required=True, help="Output directory")
        group.add_argument("--overwrite", "--clobber", help="If specified will overwrite output directory if it already exists", action="store_true", default=False)
        
        group = self.add_argument_group("Preprocessing options")
        group.add_argument("--noprune", help="Do not remove files that we don't think we need (low-res anatomical, single volume DTI, localizer)", action="store_true", default=False)
        group.add_argument("--noreorient", help="Do not re-orient files to standard orientation", action="store_true", default=False)
        group.add_argument("--nofixdti", help="Do not try to 'fix' S-I flipping in DTI files", action="store_true", default=False)
        group.add_argument("--nofixbval", help="Do not try to 'fix' BVAL ordering in DTI files", action="store_true", default=False)
        group.add_argument("--nodticpgeom", help="Do not copy the geometry of S-I flipped DTI files from the first non-flipped DTI", action="store_true", default=False)
        group.add_argument("--flipap", help="When re-orienting, flip the A-P axis for all scans. Used when rodent has been put tail-first into the bore rather than nose-first", action="store_true", default=False)
        group.add_argument("--debug", help="Enable debug output", action="store_true", default=False)
       
def preproc(categorized_files, options):
    for cat, fpaths in categorized_files.items():
        for fpath in fpaths:
            if cat == ANAT_BEST:
                os.replace(fpaths[0], os.path.join(options.output, "anat.nii.gz"))
                os.replace(sidecar(fpaths[0], "json"), os.path.join(options.output, "anat.json"))
            elif cat == DTI:
                standardize_dti(fpath)

def prune(categorized_files, options):
    if not options.noprune:
        for cat, fpaths in categorized_files.items():
            if cat not in (DTI, ANAT_BEST):
                for fpath in fpaths:
                    os.remove(fpath)
                    for ext in ("json", "bval", "bvec"):
                        fpath_sidecar = sidecar(fpath, ext)
                        if os.path.exists(fpath_sidecar):
                            os.remove(fpath_sidecar)

def fix_dti(categorized_files, options):
    if not options.nofixdti:
        if categorized_files[ANAT_BEST]:
            anat_fpath = categorized_files[ANAT_BEST][0]
            flipped_dtis, nonflipped_dtis = [], []
            for fpath in categorized_files[DTI]:
                was_flipped = fix_dti_orientation(fpath, anat_fpath)
                if not was_flipped:
                    nonflipped_dtis.append(fpath)
                else:
                    flipped_dtis.append(fpath)

            # Sometimes affine in reverse files comes out wrong
            if nonflipped_dtis and flipped_dtis and not options.nodticpgeom:
                for fpath in flipped_dtis:
                    os.system(f'fslcpgeom "{nonflipped_dtis[0]}" "{fpath}" -d')
        else:
            LOG.warn("Can't fix DTI orientation - we don't have an anatomical image")

def fix_bval_ordering(categorized_files, options):
    if not options.nofixbval:
        for cat, fpaths in categorized_files.items():
            if cat == DTI:
                for fpath in fpaths:
                    try:
                        bvals = np.loadtxt(sidecar(fpath, "bval"))
                        bvecs = np.loadtxt(sidecar(fpath, "bvec")).T
                        zero_bvecs = [idx for idx, bvec in enumerate(bvecs) if np.allclose(bvec, [0, 0, 0])]
                        num_b0s = len(zero_bvecs)
                        smallest_bvals = sorted(np.argpartition(bvals, num_b0s)[:num_b0s])
                        smallest_bvals_copy = list(smallest_bvals)
                        if zero_bvecs != smallest_bvals:
                            print(f"WARNING: Smallest BVALS do not correspond with positions of zero bvecs for {fpath}")
                            print(f"Zero bvecs at: {zero_bvecs}" % zero_bvecs)
                            print(f"Smallest bvals at: {smallest_bvals}")
                            new_bvals, current_bval = [], 0
                            for idx in range(len(bvals)):
                                if idx in zero_bvecs:
                                    new_bvals.append(bvals[smallest_bvals.pop(0)])
                                else:
                                    while current_bval in smallest_bvals_copy:
                                        current_bval += 1
                                    new_bvals.append(bvals[current_bval])
                                    current_bval += 1
                            print("Re-ordered BVALS: %s" % new_bvals)
                            with open(sidecar(fpath, "bval"), "w") as f:
                                np.savetxt(f, new_bvals, newline=" ", fmt='%f')
                    except:
                        print(f"WARNING: Failed to get bvals/bvecs from DTI: {fpath}")
                        traceback.print_exc()

def main():
    parser = ArgumentParser()
    options = parser.parse_args()
    try:
        if not os.path.isdir(options.input):
            parser.error(f"Input directory {options.input} does not exist or is not a directory")
        if os.path.exists(options.output) and not options.overwrite:
            parser.error(f"Output directory {options.output} already exists - use --overwrite to ignore")

        makedirs(options.output, exist_ok=True, clean_imgs=True)
        setup_logging(options.output, level="DEBUG" if options.debug else "INFO", save_log=True, log_stream=sys.stdout, logfile_name="preproc.log")

        # Convert data to NIFTI in standard orientation and categorize files
        convert_to_nifti(options.input, options.output)
        if not options.noreorient:
            reorient_niftis(options.output, flip_ap=options.flipap)
        categorized_files = categorize_niftis(options.output)

        # Preprocess files by category
        preproc(categorized_files, options)
        
        # Prune files we don't need
        prune(categorized_files, options)

        # Fix DTI orientation if requested
        fix_dti(categorized_files, options)
        
        # Fix DTI BVAL ordering if requested
        fix_bval_ordering(categorized_files, options)
    except Exception as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        if options.debug:
            raise
        else:
            sys.exit(1)
