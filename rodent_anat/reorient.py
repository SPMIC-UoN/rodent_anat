"""
rodent_mri: Code to re-orient data to 'standard' orientation

Note that we don't use fslreorient2std as we want to be able to apply
the method to DTI images as well which have bvecs that need similar
processing
"""
import logging
import os
import shutil

import nibabel as nib
import numpy as np

from .utils import sidecar

LOG = logging.getLogger(__name__)

def _reorient_bvec(bvec_fpath, out_fpath, dim_reorder, dim_flip):
    """
    Transform BVEC vectors from a DTI image according to the standard
    orientation changes that are being applied to the main Nifti file
    """
    bvec = np.loadtxt(bvec_fpath)
    if bvec.ndim < 2:
        bvec = bvec[:, np.newaxis]

    # Apply data transposition/flipping to BVECs
    new_bvec = np.copy(bvec)
    for idx, dim in enumerate(dim_reorder):
        new_bvec[dim, :] = bvec[idx, :]
        
    for dim in dim_flip:
        new_bvec[dim, :] = -new_bvec[dim, :]
    
    # Write out converted BVECs
    with open(out_fpath, "w") as f:
        np.savetxt(f, new_bvec, fmt='%f')

def to_std_orientation(fpath, out_fpath=None):
    """
    Convert Nifti data to standard internal orientation

    :param fpath: Path to Nifti image
    :param out_fpath: If specified, path to save output. If not specified
                      output will overwrite input

    :return: Tuple of axis reordering, axis flips, reordered data array
    """
    LOG.info(f" - Re-orienting {fpath} to standard orientation")
    nii = nib.load(fpath)
    data = nii.get_fdata()
    affine = nii.header.get_best_affine()
    LOG.debug(" - Affine:")
    LOG.debug(affine)

    transform = affine[:3, :3]
    dim_reorder, dim_flip = [], []
    absmat = np.absolute(transform)
    for dim in range(3):
        newd = np.argmax(absmat[:, dim])
        dim_reorder.append(newd)
        if transform[newd, dim] < 0:
            dim_flip.append(newd)

    LOG.debug(f" - Dimension re-order: {dim_reorder}, flip: {dim_flip}")
    if sorted(dim_reorder) != [0, 1, 2]:
        raise RuntimeError("Could not find consistent dimension re-ordering")
    
    new_data = np.copy(data)
    new_affine = np.copy(affine)

    # Re-order axes
    if new_data.ndim == 4:
        dim_transpose = list(dim_reorder) + [3]
    else:
        dim_transpose = dim_reorder
    new_data = np.transpose(new_data, dim_transpose)
    for idx, dim in enumerate(dim_reorder):
        new_affine[:, dim] = affine[:, idx]

    # Flip dimensions
    for dim in dim_flip:
        new_data = np.flip(new_data, dim)
        new_affine[:, dim] = -new_affine[:, dim]

    # Adjust origin to correct axes flips
    for dim in dim_flip:
        new_affine[:3, 3] = new_affine[:3, 3] - new_affine[:3, dim] * (new_data.shape[dim]-1)
    LOG.debug(" - New affine:")
    LOG.debug(new_affine)

    if not out_fpath:
        out_fpath = fpath
    else:
        # If moving the file elsewhere need to copy sidecar files as well (note
        # that bvec is handled separately below as it needs transformation)
        for sidecar_ext in ("json", "bval"):
            sidecar_fpath = sidecar(fpath, sidecar_ext)
            if os.path.exists(sidecar_fpath):
                out_sidecar_fpath = sidecar(out_fpath, sidecar_ext)
                shutil.copy(sidecar_fpath, out_sidecar_fpath)

    # Save main Nifti output
    nii_out = nib.Nifti1Image(new_data, header=nii.header, affine=new_affine)
    nii_out.to_filename(out_fpath)

    # Convert BVECs if present
    bvec_fpath = sidecar(fpath, "bvec")
    if os.path.exists(bvec_fpath):
        out_bvec_fpath = sidecar(out_fpath, "bvec")
        _reorient_bvec(bvec_fpath, out_bvec_fpath, dim_reorder, dim_flip)

def reorient_niftis(niftidir):
    """
    Re-orient all the Nifti files in a directory in-place
    """
    LOG.info(f"Re-orienting files in {niftidir} to standard orientation")
    for path, _dirs, files in os.walk(niftidir):
        for fname in files:
            fpath = os.path.join(path, fname)
            if ".nii" in fname:
                to_std_orientation(fpath)
