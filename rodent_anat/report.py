"""
rodent_anat: Output report
"""
import os
import logging

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

LOG = logging.getLogger(__name__)

def run(options):
    LOG.info("GENERATING REPORT")

    outdir = os.path.abspath(options.output)

    report_fname = os.path.join(outdir, "report.pdf")
    pdf = PdfPages(report_fname)

    plt.figure(figsize=(8.27,11.69))   # Standard portrait A4 sizes
    plt.suptitle(f"RODENT_ANAT: Report for {options.input}", fontsize=10, fontweight='bold')

    #self._generate(pdf)

    plt.tight_layout(h_pad=1, pad=4)
    plt.savefig(pdf, format='pdf')
    plt.close()

    pdf.close()
