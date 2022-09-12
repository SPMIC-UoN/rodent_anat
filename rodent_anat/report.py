"""
rodent_anat: Output report

Jenna's suggestions:

 - SNR
 - CNR
 - some metric to show that bias field correction reduces signal inhomogeneity 
   within and across slices
 - cross correlation with template in subject space before and after both linear 
   and non-linear registration (for non-linear it could also be a comparison 
   between mmorf and ANTs). This would allow us to confirm that the registration 
   improves alignment
 - cross correlation between tissue maps generated by segmentation and the 
   corresponding probability maps from the template in subject space (this is 
   only relevant to where we use the maps as priors and run fast-based 
   segmentation).

I also think it would be great to include the sanity check snapshots generated 
by slicer throughout the pipeline so that can visually confirm what has happened 
with each step.
"""
import os
import logging
import argparse
import sys

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from . import utils

LOG = logging.getLogger(__name__)

def run(options):
    Report(options).run()

class Report:

    def __init__(self, options):
        self.options = options
        self.outdir = os.path.abspath(options.output)
        self.qcdir = os.path.join(self.outdir, "qc")
        self.report_fname = os.path.join(self.outdir, "report.pdf")
        self.pdf = PdfPages(self.report_fname)

    def _show_table(self, ax, table_title, table_content, table_colours=None):
        """
        Write a table to the PDF
        """
        ax.axis('off')
        ax.axis('tight')
        ax.set_title(table_title, fontsize=12, fontweight='bold',loc='left')
        c1len = max([len(str(c[0])) for c in table_content])
        c2len = max([len(str(c[1])) for c in table_content])
        col_prop = c1len / (c1len+c2len)
        tb = ax.table(
            cellText=table_content, 
            cellColours=table_colours,
            loc='upper center',
            cellLoc='left',
            colWidths=[col_prop, 1-col_prop],
        )
        tb.auto_set_font_size(True)
        tb.scale(1, 2)

    def _slicer_image(self, ax, title, fname):
        img = matplotlib.image.imread(os.path.join(self.qcdir, fname))
        im = ax.imshow(img, interpolation='none')
        #plt.colorbar(im, ax=ax)
        ax.grid(False)
        ax.axis('off')
        ax.set_title(title)

    def _newpage(self, first=False):
      if not first:
          plt.tight_layout(h_pad=1, pad=4)
          plt.savefig(self.pdf, format='pdf')
          plt.close()
      plt.figure(figsize=(8.27,11.69))   # Standard portrait A4 sizes
      plt.suptitle(f"RODENT_ANAT: Report for {self.options.input}", fontsize=10, fontweight='bold')

    def run(self):
        LOG.info("START: GENERATING REPORT")
        self._newpage(first=True)

        qc_measures = []
        qc_measures.append(["CNR", "1.23"])
        qc_measures.append(["SNR", "4.56"])
        qc_measures.append(["Signal inhomogenaity (non bias corrected)", "7.89"])
        qc_measures.append(["Signal inhomogenaity (bias corrected)", "1.23"])
        qc_measures.append(["Cross-correlation with template (linear)", "4.56"])
        qc_measures.append(["Cross-correlation with template (nonlinear)", "7.89"])
        self._show_table(plt.subplot(3, 1, 1), "QC measures", qc_measures)

        self._slicer_image(plt.subplot(3, 1, 2), "Linear registration", "reg_linear.png")
        self._slicer_image(plt.subplot(3, 1, 3), "Non-linear registration", "reg_nonlin.png")
        self._newpage()
        
        self._slicer_image(plt.subplot(3, 1, 1), "Bias-corrected", "biascorr.png")
        self._slicer_image(plt.subplot(3, 1, 2), "Non-bias corrected", "nobiascorr.png")
        self._newpage()

        self._slicer_image(plt.subplot(4, 1, 1), "Brain-extraction (linear)", "skull_strip_linear.png")
        self._slicer_image(plt.subplot(4, 1, 2), "Brain-extraction (nonlinear)", "skull_strip_nonlin.png")
        self._slicer_image(plt.subplot(4, 1, 3), "Brain-extraction (linear, template space)", "skull_strip_std_space_linear.png")
        self._slicer_image(plt.subplot(4, 1, 4), "Brain-extraction (nonlinear, template space)", "skull_strip_std_space_nonlin.png")
        self._newpage()
        
        self.pdf.close()

        LOG.info("DONE: GENERATING REPORT")

class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, **kwargs):
        argparse.ArgumentParser.__init__(self, prog="reorient2std", add_help=True, **kwargs)
        self.add_argument("output", help="Directory containing output of rodent_anat")
        self.add_argument("--report-fname", help="Output filename", default="report.pdf")
        self.add_argument("--debug", action="store_true", default=False, help="Enable debug output")
        self.add_argument("--template", help="Path to directory containing SIGMA template files (default=$HOME/SIGMA_template)")

def main():
    parser = ArgumentParser()
    options = parser.parse_args()
    try:
        utils.setup_logging(level="DEBUG" if options.debug else "WARN", log_stream=sys.stdout)
        options.input = os.path.basename(options.output)
        Report(options).run()
    except Exception as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        if options.debug:
            raise
        else:
            sys.exit(1)
