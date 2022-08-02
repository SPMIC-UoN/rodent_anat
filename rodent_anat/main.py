"""
rodent_anat: Definition of command line interface and main script entry point
"""
import argparse
import os
import sys
from pathlib import Path

from . import utils, anat_pipeline, report

class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, **kwargs):
        argparse.ArgumentParser.__init__(self, prog="rodent_anat", add_help=True, **kwargs)

        group = self.add_argument_group("Input/output Options")
        group.add_argument("-i", "--input", required=True, help="T2 weighted structural image - should be isotropic")
        group.add_argument("-o", "--output", help="Basename of directory for output (default is input image basename followed by .anat)")
        group.add_argument("--overwrite", "--clobber", help="Overwrite existing output directory", action="store_true", default=False)
        group.add_argument("--nocleanup", help="Do not remove intermediate files", action="store_true", default=False)
        group.add_argument("--noreport", help="Do not generate report", action="store_true", default=False)
        
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
        group.add_argument("--nonlinreg", help="Non-linear registration method", choices=["ants", "mmorf"], default="mmorf")
        group.add_argument("--template", help="Path to directory containing SIGMA template files (default=$HOME/SIGMA_template)")
        group.add_argument("--debug", help="Enable debug output", action="store_true", default=False)

        group = self.add_argument_group("Pipeline dependencies")
        group.add_argument("--antsdir", help="Path to ANTs executable directory", default="/usr/local/ANTsX/install/bin")
        group.add_argument("--c3ddir", help="Path to c3D executable directory", default="/usr/local/c3d/bin")
        group.add_argument("--mmorfdir", help="Path to MMORF config and Singularity image directory (default=$HOME/mmorf)")

def main():
    parser = ArgumentParser()
    options = parser.parse_args()
    try:
        if options.strongbias and options.weakbias:
            parser.error("Can't specify --strongbias and --weakbias at the same time")
        elif not options.strongbias:
            options.weakbias = True

        if not options.output:
            inpath = os.path.abspath(options.input)
            inpath = inpath[:inpath.index(".nii")]
            options.output = inpath + ".anat"

        if not options.template:
            options.template = os.path.join(Path.home(), "SIGMA_template")

        if not options.mmorfdir:
            options.mmorfdir = os.path.join(Path.home(), "mmorf")

        if os.path.exists(options.output) and not options.overwrite:
            parser.error(f"Output directory {options.output} already exists - use --overwrite to ignore")

        utils.makedirs(options.output, True)
        utils.setup_logging(options.output, level="DEBUG" if options.debug else "INFO", save_log=True, log_stream=sys.stdout, logfile_name="rodent_anat.log")

        anat_pipeline.run(options)
        if not options.noreport:
            report.run(options)
    except Exception as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        if options.debug:
            raise
        else:
            sys.exit(1)
