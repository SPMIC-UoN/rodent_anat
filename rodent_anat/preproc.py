"""
rodent_anat: Command line tool to preprocess raw Bruker data
"""
import argparse
import os
import logging
import sys

from .utils import makedirs, setup_logging, sidecar
from .nifti_convert import convert_to_nifti
from .categorize import categorize_niftis, ANAT_BEST, DTI
from .dti import fix_dti_orientation, standardize_dti
from .reorient import reorient_niftis

LOG = logging.getLogger(__name__)

class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, **kwargs):
        argparse.ArgumentParser.__init__(self, prog="bruker_preproc", add_help=True, **kwargs)

        group = self.add_argument_group("Input/output Options")
        group.add_argument("-i", "--input", help="Directory containing raw Bruker data", default="")
        group.add_argument("-o", "--output", help="Output directory", default="")
        group.add_argument("--overwrite", help="If specified will overwrite output directory if it already exists", action="store_true", default=False)
        
        group = self.add_argument_group("Preprocessing options")
        group.add_argument("--noprune", help="Do not remove files that we don't think we need (low-res anatomical, single volume DTI, localizer)", action="store_true", default=False)
        group.add_argument("--noreorient", help="Do not re-orient files to standard orientation", action="store_true", default=False)
        group.add_argument("--nofixdti", help="Do not try to 'fix' S-I flipping in DTI files", action="store_true", default=False)
        group.add_argument("--nodticpgeom", help="Do not copy the geometry of S-I flipped DTI files from the first non-flipped DTI", action="store_true", default=False)
        group.add_argument("--debug", help="Enable debug output", action="store_true", default=False)
       
def main():
    options = ArgumentParser().parse_args()
    if not os.path.isdir(options.input):
        raise ValueError(f"Input directory {options.input} does not exist or is not a directory")
    if not options.output:
        raise ValueError(f"Output directory not specified")
    if os.path.exists(options.output) and not options.overwrite:
        raise ValueError(f"Output directory {options.output} already exists - use --overwrite to ignore")

    makedirs(options.output, True)
    setup_logging(options.output, level="DEBUG" if options.debug else "INFO", save_log=True, log_stream=sys.stdout, logfile_name="preproc.log")

    convert_to_nifti(options.input, options.output)

    if not options.noreorient:
        reorient_niftis(options.output)

    # Handle files by category
    categories = categorize_niftis(options.output)
    for cat, fpaths in categories.items():
        for fpath in fpaths:
            if cat == ANAT_BEST:
                print("Renaming to anat: " + fpaths[0])
                os.rename(fpaths[0], os.path.join(options.output, "anat.nii.gz"))
                os.rename(sidecar(fpaths[0], "json"), os.path.join(options.output, "anat.json"))
            elif cat == DTI:
                standardize_dti(fpath)

    # Prune files we don't need
    if not options.noprune:
        for cat, fpaths in categories.items():
            if cat not in (DTI, ANAT_BEST):
                for fpath in fpaths:
                    os.remove(fpath)

    # Fix DTI orientation if requested
    if not options.nofixdti:
        if categories[ANAT_BEST]:
            anat_fpath = categories[ANAT_BEST][0]
            flipped_dtis, nonflipped_dtis = [], []
            for fpath in categories[DTI]:
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
