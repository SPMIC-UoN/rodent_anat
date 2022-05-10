"""
rodent_mri: Useful utility functions
"""
from contextlib import contextmanager
import logging
import os
import tempfile

import nibabel as nib

LOG = logging.getLogger(__name__)

@contextmanager
def working_dir(path):
    """
    Change directory as a context manager
    """
    origin = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(origin)

def sidecar(img_fpath, ext):
    """
    :param img_fpath: Path to Nifti image with .nii or .nii.gz extension
    :param ext: Sidecar extension without .

    :return: Path to sidecar of Nifti file with given extension
    """
    return img_fpath[:img_fpath.index(".nii")] + "." + ext

def num_vols(nii):
    """
    :return: Number of volumes in Nifti dataset
    """
    if isinstance(nii, str):
        nii = nib.load(nii)
    return 1 if nii.ndim == 3 else nii.shape[3]

def movedir(path, dir):
    """
    :return: Path to file with same name but different directory
    """
    return os.path.join(dir, os.path.basename(path))

def orient_cost(img_fname, ref_fname):
    """
    Get the cost of aligning two images by translation alone
    
    The idea is that this will be higher when one image is incorrectly
    flipped wrt the other
    """
    if not os.path.exists(ref_fname):
        # No reference
        return 0

    with tempfile.TemporaryDirectory() as d:
        if "FSLDIR" in os.environ:
            fsldir = os.environ["FSLDIR"]
        else:
            raise RuntimeError("FSLDIR is not set")
        os.system(f'fslroi {img_fname} vol1 0 1')
        os.system(f'flirt -in vol1 -ref "{ref_fname}" -schedule {fsldir}/etc/flirtsch/xyztrans.sch -omat xyztrans.mat >regout 2>reg_stderr')
        os.system(f'flirt -in vol1 -ref "{ref_fname}" -schedule {fsldir}/etc/flirtsch/measurecost1.sch -init xyztrans.mat >costout 2>cost_stderr')
        with open("costout", "r") as f:
            for line in f:
                cost = float(line.split()[0])
                LOG.debug(f"Alignment of {img_fname} and {ref_fname}: cost={cost}")
                return cost

def makedirs(dpath, exist_ok=False):
    """
    Make directories, optionally ignoring them if they already exist
    """
    try:
        os.makedirs(dpath)
    except OSError as exc:
        import errno
        if not exist_ok or exc.errno != errno.EEXIST:
            raise

def setup_logging(outdir=".", logfile_name="logfile", **kwargs):
    """
    Set the log level, formatters and output streams for the logging output

    By default this goes to <outdir>/logfile at level INFO
    """

    # Set log level on the root logger to allow for the possibility of 
    # debug logging on individual loggers
    level = kwargs.get("log_level", "info")
    if not level:
        level = "info"
    level = getattr(logging, level.upper(), logging.INFO)
    logging.getLogger().setLevel(level)

    if outdir and kwargs.get("save_log", False):
        # Send the log to an output logfile
        makedirs(outdir, True)
        logfile = os.path.join(outdir, logfile_name)
        logging.basicConfig(filename=logfile, filemode="w", level=level)

    if kwargs.get("log_stream", None) is not None:
        # Can also supply a stream to send log output to as well (e.g. sys.stdout)
        extra_handler = logging.StreamHandler(kwargs["log_stream"])
        extra_handler.setFormatter(logging.Formatter('%(levelname)s : %(message)s'))
        logging.getLogger().addHandler(extra_handler)
