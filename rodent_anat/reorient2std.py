"""
rodent_anat: Command line tool intended to replicate fslreorient2std but handle bvec files as well
"""
import argparse
import sys

from . import reorient, utils

class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, **kwargs):
        argparse.ArgumentParser.__init__(self, prog="reorient2std", add_help=True, **kwargs)
        self.add_argument("input", help="Input image")
        self.add_argument("output", help="Output image")
        self.add_argument("--convention", choices=["neuro", "radio"], default="radio", help="Orientation convention for internal storage: determinant of SFORM/QFORM is <0 for radio, >0 for neuro")
        self.add_argument("--debug", action="store_true", default=False, help="Enable debug output")

def main():
    parser = ArgumentParser()
    options = parser.parse_args()
    try:
        utils.setup_logging(options.output, level="DEBUG" if options.debug else "INFO", save_log=False, log_stream=sys.stdout)
        reorient.to_std_orientation(options.input, options.output, convention=reorient.NEURO if options.convention == "neuro" else reorient.RADIO)
    except Exception as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        if options.debug:
            raise
        else:
            sys.exit(1)
