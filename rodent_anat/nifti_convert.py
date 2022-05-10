"""
rodent_anat: Support code for conversion of Bruker raw data to Nifti

Currently uses the brkraw tool
"""
import os
import logging

LOG = logging.getLogger(__name__)

def convert_to_nifti(indir, outdir):
    """
    Convert bruker data to Nifti

    :param indir: Path to directory containing raw Bruker data
    :param outdir: Path to directory for output Nifti files
    """
    LOG.info(f"Converting data in {indir} to BIDS using brkraw")
    subj = os.path.basename(os.path.dirname(indir))
    for fname in os.listdir(indir):
        fpath = os.path.join(indir, fname)
        if os.path.isdir(fpath):
            try:
                scanid = int(fname)
            except ValueError:
                continue

            LOG.info(f" - Converting scan: {scanid}")
            status = os.system(f"brkraw tonii {indir} -b -o {outdir}/{subj} -s {scanid} 2>stderr >stdout")
            if status != 0:
                LOG.warn(f" - Scan {scanid}: conversion FAILED")
