"""
rodent_anat: Categorization of Nifti files by type
"""
from collections import defaultdict
import logging
import os

import nibabel as nib

from .utils import sidecar

LOCALIZER = "localizer"
ANAT = "anat"
ANAT_BEST = "anat_best"
DTI = "dti"
DTI_SINGLEVOL = "dti_singlevol"

LOG = logging.getLogger(__name__)

def categorize_niftis(niftidir):
    """
    Try to categorize the different types of Nifti files in a directory
    """
    LOG.info(f"Categorizing Nifti files in {niftidir}")
    categories = defaultdict(list)
    for path, _dirs, files in os.walk(niftidir):
        for fname in files:
            fpath = os.path.join(path, fname)
            if ".nii" not in fname:
                continue
            if not os.path.exists(sidecar(fpath, ".json")):
                LOG.warn(f"Ignoring Nifti file {fname} without json sidecar")
                continue

            if "localize" in fname.lower():
                cat = LOCALIZER
            elif os.path.exists(sidecar(fpath, ".bvec")):
                if num_vols(nii) == 1:
                    cat = DTI_SINGLEVOL
                else:
                    cat = DTI
            else:
                cat = ANAT
            
            LOG.debug(f"Found Nifti file {fname} categorized as {cat}")
            categories[cat].append(fpath)

    # Idenfity the anatomical file with the best resolution
    best_anat_resolution = 1000
    for fpath in categories[ANAT]:
        nii = nib.load(fpath)
        anat_resolution = np.mean(np.diag(nii.affine))
        if anat_resolution < best_anat_resolution:
            categories[ANAT_BEST] = [fpath]

    return categories
