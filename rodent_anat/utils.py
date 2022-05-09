"""
rodent_mri: Useful utility functions
"""
import os

import nibabel as nib

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
                LOG.info(f"Alignment of {img_fname} and {ref_fname}: cost={cost}")
                return cost
