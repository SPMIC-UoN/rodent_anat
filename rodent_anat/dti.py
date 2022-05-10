"""
rodent_anat: Utilities for handling DTI files
"""
import logging
import os
import shutil
import tempfile

import numpy as np
import nibabel as nib

from .utils import sidecar, num_vols, orient_cost

LOG = logging.getLogger(__name__)

def _flip_dti(dti_fpath, flipped_fpath):
    """
    Create an S-I axis flipped copy of a DTI file 
    """
    nii = nib.load(dti_fpath)
    data = nii.get_fdata()
    flipped_data = np.flip(data, 2)
    nii_out = nib.Nifti1Image(flipped_data, header=nii.header, affine=nii.header.get_best_affine())
    nii_out.to_filename(flipped_fpath)
    
    bvec_fname = sidecar(dti_fpath, "bvec")
    flipped_bvec_fname = sidecar(flipped_fpath, "bvec")
    bvec = np.loadtxt(bvec_fname)
    if bvec.ndim < 2:
        bvec = bvec[:, np.newaxis]
    flipped_bvec = np.copy(bvec)
    flipped_bvec[2, :] = -flipped_bvec[2, :]
    with open(flipped_bvec_fname, "w") as f:
        np.savetxt(f, flipped_bvec, fmt='%f')

def standardize_dti(fpath):
    """
    Standardize the DTI metadata by rounding the BVALS and padding BVALs and BVECs
    to match the number of repeats
    """
    bvec_fpath = sidecar(fpath, "bvec")
    bval_fpath = sidecar(fpath, "bval")
    if not os.path.exists(bval_fpath) or not os.path.exists(bvec_fpath):
        print(bvec_fpath, bval_fpath, os.path.exists(bval_fpath), os.path.exists(bvec_fpath))
        raise RuntimeError(f"{fpath}: was expecting DTI file but either BVEC or BVAL files are missing")

    bvec = np.loadtxt(bvec_fpath)
    if bvec.ndim < 2:
        bvec = bvec[:, np.newaxis]
    bval = np.atleast_1d(np.loadtxt(bval_fpath, dtype=float))

    # Round BVALs to nearest 1000
    bval = np.round(bval, -3)

    # Handle repeats in BVALs and BVECs
    n_bvecs = bvec.shape[1]
    nvols = num_vols(fpath)
    if nvols % n_bvecs != 0:
        raise RuntimeError("Data shape is not divisible by number of bvecs")
    num_rpts = int(nvols / n_bvecs)

    LOG.info(f'{fpath}: Expanding bvecs and bvals for {num_rpts} repeats')
    new_bvec = np.tile(bvec, num_rpts)
    new_bval = np.tile(bval, num_rpts)

    # Write out converted files
    with open(bvec_fpath, "w") as f:
        np.savetxt(f, new_bvec, fmt='%f')

    with open(bval_fpath, "w") as f:
        np.savetxt(f, new_bval, newline=" ", fmt='%f')

def fix_dti_orientation(fpath, anat_fpath, use_fname=True):
    """
    Fix the orientation of DTI files which have incorrect S-I flipping

    :param fpath: Path to DTI image
    :param fpath_anat: Path to anatomical image
    :param use_fname: If True will do S-I flipping if DTI filename suggests it is phase-reversed
    """
    with tempfile.TemporaryDirectory() as d:
        LOG.info(f"Testing {fpath} for fwd/rev orientation")
        fwd = os.path.join(d, "fwd.nii.gz") # FIXME extension
        rev = os.path.join(d, "rev.nii.gz")

        shutil.copy(fpath, fwd)
        _flip_dti(fpath, rev)
        use_rev = False

        if use_fname and "reverse" in os.path.basename(fpath).lower():
            LOG.info(f"DTI file {fpath} has a file name suggesting it is phase-reversed - will perform S-I flip")
            use_rev = True
        else:
            cost_fwd = orient_cost(fwd, anat_fpath)
            cost_rev = orient_cost(rev, anat_fpath)
            if cost_rev < cost_fwd:
                LOG.info(f"DTI file {fpath} matches the anatomical image better if it is S-I flipped")
                use_rev = True
        if use_rev:
            shutil.copy(rev, fpath)
            shutil.copy(sidecar(rev, "bvec"), sidecar(fpath, "bvec"))
