import argparse

class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, **kwargs):
        argparse.ArgumentParser.__init__(self, prog="rodent_anat", add_help=True, **kwargs)

        group = self.add_argument_group("Input/output Options")
        group.add_argument("-i", help="Structural image")
        group.add_argument("-t", help="Type of image (T1 T2 or PD - default is T1)")
        #group.add_argument("-d", help="Existing .anat directory where this script will be run in place")
        group.add_argument("-o", help="Basename of directory for output (default is input image basename followed by .anat)")
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

def main():
    args = ArgumentParser().parse_args()
    if args.strongbias and args.weakbias:
        raise ValueError("Can't specify --strongbias and --weakbias at the same time")
    elif not args.strongbias:
        args.weakbias = True

    print(vars(args))
