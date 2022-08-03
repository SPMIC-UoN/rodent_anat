"""
rodent_anat: Output report
"""
import os
import logging

import matplotlib
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

    img = matplotlib.image.imread("thingy.png")
    ax = plt.subplot(1, 1, 1)
    im = ax.imshow(img, interpolation='none', cmap="gray", vmin = 0, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.grid(False)
    ax.axis('off')
    ax.set_title('')

    plt.tight_layout(h_pad=1, pad=4)
    plt.savefig(pdf, format='pdf')
    plt.close()

    pdf.close()
