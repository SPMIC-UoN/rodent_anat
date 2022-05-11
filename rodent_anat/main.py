"""
rodent_anat: Definition of command line interface and main script entry point
"""
import argparse
import os
import sys

from .utils import makedirs, setup_logging
from .pipeline import run_anat_pipeline

class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, **kwargs):
        argparse.ArgumentParser.__init__(self, prog="rodent_anat", add_help=True, **kwargs)

        group = self.add_argument_group("Input/output Options")
        group.add_argument("-i", "--input", help="T2 weighted structural image - should be isotropic")
        #group.add_argument("-t", help="Type of image (T1 T2 or PD - default is T1)")
        #group.add_argument("-d", help="Existing .anat directory where this script will be run in place")
        group.add_argument("-o", "--output", help="Basename of directory for output (default is input image basename followed by .anat)")
        group.add_argument("--clobber", help="Type of average to report in iteration logs (mean or median)", action="store_true", default=False)
        group.add_argument("--nocleanup", help="Do not remove intermediate files", action="store_true", default=False)
        
        group = self.add_argument_group("Pipeline options")
        group.add_argument("--strongbias", help="Used for images with very strong bias fields", action="store_true", default=False)
        group.add_argument("--weakbias", help="Used for images with smoother, more typical, bias fields (default setting)", action="store_true", default=False)
        group.add_argument("--nocrop", help="Turn off step that does automated cropping", action="store_true", default=False)
        group.add_argument("--noreorient", help="Turn off step that does reorientation to standard", action="store_true", default=False)
        group.add_argument("--nobias", help="Turn off steps that do bias field correction", action="store_true", default=False)
        group.add_argument("--noreg", help="Turn off steps that do registration to standard", action="store_true", default=False)
        group.add_argument("--nononlinreg", help="Turn off step that does non-linear registration", action="store_true", default=False)
        group.add_argument("--noseg", help="Turn off step that does tissue-type segmentation ", action="store_true", default=False)
        group.add_argument("--nosearch", help="Specify that linear registration uses the -nosearch option", action="store_true", default=False)
        group.add_argument("-s", "--biassmooth", help="Specify the value for bias field smoothing (the -l option in FAST)", type=int, default=10)
        group.add_argument("--nonlinreg", help="Non-linear registration method", choices=["ants", "mmorf"])
        group.add_argument("--template", help="Path to template file if not using default SIGMA template")
        group.add_argument("--debug", help="Enable debug output", action="store_true", default=False)

        group = self.add_argument_group("Pipeline dependencies")
        group.add_argument("--antspath", help="Path to ANTs executable", default="/usr/local/ANTsX/install/bin")
        group.add_argument("--c3dpath", help="Path to c3D executables", default="/usr/local/c3d/bin")
        group.add_argument("--mmorfdir", help="Path to MMORF config and Singularity image", default="/home/bbzmsc/mmorf")

def main():
    options = ArgumentParser().parse_args()
    if options.strongbias and options.weakbias:
        raise ValueError("Can't specify --strongbias and --weakbias at the same time")
    elif not options.strongbias:
        options.weakbias = True

    if not options.template:
        options.template = "/home/bbzmsc/SIGMA_template"

    if not options.input:
        raise ValueError(f"Input image not specified")
    if not options.output:
        options.output = os.path.basename(options.input) + ".anat"

    if os.path.exists(options.output) and not options.clobber:
        raise ValueError(f"Output directory {options.output} already exists - use --clobber to ignore")

    makedirs(options.output, True)
    setup_logging(options.output, level="DEBUG" if options.debug else "INFO", save_log=True, log_stream=sys.stdout, logfile_name="rodent_anat.log")

    run_anat_pipeline(options)
