"""
rodent_anat: Main anatomical pipeline

Description: Main script with all the processing for rodent T2
 1. Reorient rodent scan so matches template brain
 2. Generate affine transform to register T2 volume to template brain
 3. Invert affine transform and use to take template brain mask to native space
 3. Perform 'brain extraction' using brain mask in native space
 4. Use FAST to correct for distortions caused by bias field (requires brain extracted data)
 5. Refine linear registration step for T2 to template brain using bias field corrected skull stripped data
 (6a. Use the affine transformed data to initialise the registration step using ANTs, convert ANTs warps to FSL format - note, need to first combine fsl .mat with ANTs .mat before combining .mat with warp)
 (6b. Use the affine transformation to initialise the non-linear registration step using mmorf, will first need to generate config)
 7. Revise brain mask after non-linear registration to refine 'brain extraction' step
 8. Segment brain into tissue classes, assess performance by comparing with registered tissue masks

Dependencies
 1. FSL (https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/)
 2. ANTs (https://github.com/ANTsX/ANTs) - this is optional
 3. If using ANTs: C3D (http://www.itksnap.org/pmwiki/pmwiki.php?n=Downloads.C3D) - used to convert ANTs warps so that compatible with FSL (note that these are not diffeomorphic)
 4. mmorf (https://git.fmrib.ox.ac.uk/flange/mmorf_beta?web=1&wdLOR=c33787EB2-D6A8-3B41-BB61-5EF9FF5B073F#scalar-image-options)
 5. If using mmorf: singularity and Nvidia GPU

Standard templates:

Templates are derived from Barriere et al., 2019, Nat Commun, see: https://doi.org/10.1038/s41467-019-13575-7
The olfactory bulb has been removed and the binary mask has been dilated using fslmaths -ero
have also used fslorient2std

 Authors: Jenna Hanmer, Matteo Bastiani & Stamatios Sotiropoulos
"""

import logging
import os
import shutil

import nibabel as nib

from fsl.data.image import Image, defaultExt
import fsl.wrappers as fsl
from fsl.utils.imcp import immv

from .reorient import to_std_orientation
from .utils import working_dir, makedirs

LOG = logging.getLogger(__name__)

def run_anat_pipeline(options):
    LOG.info("START: rodent_anat")
    LOG.info("Step 1: housekeeping, error checking and reorientation")

    outdir = os.path.abspath(options.output)

    # Templates FIXME need better system for customization
    templdir=options.template
    templ = os.path.join(templdir, "SIGMA_ExVivo_Brain_Template.nii.gz")
    templmask = os.path.join(templdir, "SIGMA_ExVivo_Brain_Mask_No_OlfBulb.nii.gz")
    templmaskdil = os.path.join(templdir, "SIGMA_ExVivo_Brain_Mask_No_OlfBulb.nii.gz")
    #templmaskdil = os.path.join(templdir, "SIGMA_ExVivo_Brain_Mask_No_OlfBulb_dil.nii.gz")
    templbrain = os.path.join(templdir, "SIGMA_ExVivo_Brain_Template_Masked_No_OlfBulb.nii.gz")
    map_GM = os.path.join(templdir, "SIGMA_ExVivo_GM_No_OlfBulb.nii.gz")
    map_WM = os.path.join(templdir, "SIGMA_ExVivo_WM_No_OlfBulb.nii.gz")
    map_CSF = os.path.join(templdir, "SIGMA_ExVivo_CSF_No_OlfBulb.nii.gz")

    # Check that T2 is an image
    T2 = Image(options.input)
    
    with working_dir(outdir):
        if options.noreorient:
            LOG.info(" - Copying T2 to output directory")
            T2.save("T2_reorient.nii.gz")
        else:
            to_std_orientation(T2.dataSource, "T2_reorient.nii.gz")

        LOG.info("Step 2: Performing initial brain extraction and registration to standard template")

        LOG.info(" - Performing linear registration of T2 to brain-only template")
        fsl.flirt("T2_reorient", templbrain, out="T2_to_templ_linear_tmp", omat="T2_to_templ_linear_tmp.mat") # -dof 12

        # Invert this transform
        fsl.invxfm("T2_to_templ_linear_tmp.mat", "templ_to_T2_linear_tmp.mat")

        LOG.info(" - Transforming dilated template brain mask to T2 native space")
        fsl.applyxfm(templmaskdil, "T2_reorient", "templ_to_T2_linear_tmp.mat", interp="nearestneighbour", out="T2_mask_dil")

        LOG.info(" - Removing skull from T2")
        fsl.fslmaths("T2_reorient").mul("T2_mask_dil").run("T2_brain_tmp")

        LOG.info(" - Transforming brain extracted T2 into standard template space")
        fsl.applyxfm("T2_brain_tmp", templbrain, "T2_to_templ_linear_tmp.mat", out="T2_brain_to_templ_tmp", interp="trilinear")

        # Check that the linear registration of the T2 brain to the standard template was successful
        fsl.slicer("T2_brain_to_templ_tmp", templbrain, a="brain_reorient.png")

        # Check that the brain masking was successful
        fsl.slicer("T2_reorient", "T2_mask_dil", a="skull_strip_tmp.png")

        LOG.info("Step 3: correct bias-field in skull-stripped T2 volume")

        makedirs("fast", exist_ok=True)
        LOG.info(f" - Running FAST to estimate bias field: smoothing (FWHM) = {options.biassmooth}mm")
        fsl.fast("T2_brain_tmp", out="fast/T2_brain_tmp", b=True, B=True, nopve=True, lowpass=options.biassmooth, type=2, segments=False)

        # # Rename the restored/undistorted (ud) brain image
        immv("fast/T2_brain_tmp_restore", "fast/T2_brain_tmp_ud", overwrite=True)

        # # Check that the bias correction was successful, keep intensity range constant
        fsl.slicer("fast/T2_brain_tmp_ud", i="0 1", a="bias_corr.png")
        fsl.slicer("T2_brain_tmp", i="0 1", a="no_bias_corr.png")

        if not options.nobias:
            LOG.info(" - Using estimated bias field to correct re-oriented T2 volume")
            fsl.fslmaths("T2_reorient").div("fast/T2_brain_tmp_bias").run("T2_reorient_ud")
        else:
            LOG.info(" - Bias field correction disabled")

        # # Clean up directory
        # rm ${fastdir}/T2_brain_tmp_seg*

        LOG.info("Step 4: Refine the linear registration using the bias corrected brain")

        LOG.info(" - Generate the affine transform from the bias-corrected brain to the standard brain-only template")
        fsl.flirt("fast/T2_brain_tmp_ud", templbrain, out="T2_to_templ_linear", omat="T2_to_templ_linear.mat")

        # Invert T2_to_templ_linear.mat
        fsl.invxfm("T2_to_templ_linear.mat", "templ_to_T2_linear.mat")

        # Check that the refined linear registration was successful
        fsl.slicer("T2_to_templ_linear", templbrain, i="0 1", a="linear.png")

        LOG.info(" - Comparing success of linear registration on skull-stripped data with and without bias correction")
        fsl.flirt("T2_brain_tmp", templbrain, out="T2_to_templ_linear_biased", dof=12)
        fsl.slicer("T2_to_templ_linear_biased", templbrain, a="linear_biased.png", i="0 1")

        LOG.info("Step 5: Perform nonlinear registration to map T2 to the standard template")

        # # 1.ANTs
        if options.nonlinreg == "ants":
            LOG.info(" - Running nonlinear registration using ANTs")
            makedirs("ANTs", exist_ok=True)

            # Run the script antsRegistrationSyN.sh using brain-extracted bias-corrected scan and brain-only template, default is double precision, 
            # use histogram matching (-j) because template and scan are the from the same modality, use mask (-x) for template so that exclude the
            # olfactory bulb, have chosen dilated version of mask
            # Note: antsRegistrationSyN actually includes optimisation of rigid, affine and SyN transformations
            # Note: antsRegistrationSyN works best on skull-stripped, bias corrected data
            # see: https://github.com/ANTsX/ANTs/wiki/Anatomy-of-an-antsRegistration-call
            os.system(f"{options.antspath}/antsRegistrationSyN.sh -d 3 -f {templbrain} -m T2_to_templ_linear.nii.gz -j 1 -o ANTs/T2_to_templ_ANTs_ -x {templmaskdil} |tee ANTs/ants.log")

            # Check that this is working correctly
            fsl.slicer("ANTs/T2_to_templ_ANTs_Warped", templbrain, a="ANTs/ANTS_tmp.png")
                
            LOG.info(" - Converting ANTs warps so that they are compatible with FSL (ITK transform (RAS) matrix to FSL xfm)")
            os.system(f"{options.c3dpath}/c3d_affine_tool -ref {templbrain} -src T2_to_templ_linear.nii.gz -itk ANTs/T2_to_templ_ANTs_0GenericAffine.mat -ras2fsl -o ANTs/ANTs_T2_to_templ_affine_flirt.mat")

            LOG.info(" - Performing multicomponent split (-mcs)")
            os.system(f"{options.c3dpath}/c3d -mcs ANTs/T2_to_templ_ANTs_1Warp.nii.gz -oo ANTs/wx.nii.gz ANTs/wy.nii.gz ANTs/wz.nii.gz")

            LOG.info(" - Flipping warp")
            fsl.fslmaths("ANTs/wy").mul(-1).run("ANTs/i_wy")

            LOG.info(" - Concatenating components to give whole warp")
            # FIXME
            os.system("fslmerge -t ANTs/ANTs_T2_to_templ_warp_fnirt ANTs/wx ANTs/i_wy ANTs/wz")

            makedirs("ANTs/xfms", exist_ok=True)
                    
            LOG.info(" - Combining linear transform generated by ANTs with linear transform generated in step 4")
            fsl.concatxfm("ANTs/ANTs_T2_to_templ_affine_flirt.mat", "T2_to_templ_linear.mat", "ANTs/T2_to_templ_linear_x2.mat")
            # ${FSLDIR}/bin/convert_xfm -omat ${ANTsdir}/T2_to_templ_linear_x2.mat -concat ${ANTsdir}/ANTs_T2_to_templ_affine_flirt.mat ${anatdir}/T2_to_templ_linear.mat

            LOG.info(" - Combining linear transform generated in previous step with the warp generated by ANTs")
            fsl.convertwarp(ref=templbrain, premat="ANTs/T2_to_templ_linear_x2.mat", warp1="ANTs/ANTs_T2_to_templ_warp_fnirt", out="ANTs/xfms/T2_to_templ_warp")
            # ${FSLDIR}/bin/convertwarp --ref=${tempBrain} --premat=${ANTsdir}/T2_to_templ_linear_x2.mat --warp1=${ANTsdir}/ANTs_T2_to_templ_warp_fnirt --out=${ANTsdir}/xfms/T2_to_templ_warp
            # # ${FSLDIR}/bin/convertwarp --ref=${tempBrain} --premat=${ANTsdir}/ANTs_T2_to_templ_affine_flirt.mat --warp1=${ANTsdir}/ANTs_T2_to_templ_warp_fnirt --out=${ANTsdir}/xfms/T2_to_templ_warp

            LOG.info(" - Generating inverse of warp")
            fsl.invwarp("ANTs/xfms/T2_to_templ_warp", "T2_to_templ_linear", out="ANTs/xfms/templ_to_T2_warp")
            # ${FSLDIR}/bin/invwarp -w ${ANTsdir}/xfms/T2_to_templ_warp -o ${ANTsdir}/xfms/templ_to_T2_warp -r ${anatdir}/T2_to_templ_linear.nii.gz

        elif options.nonlinreg == "mmorf":
            LOG.info(" - Running nonlinear registration using MMORF")

            LOG.info(" - Creating MMORF config file")
            makedirs("mmorf", exist_ok=True)
            with open("mmorf/mmorf_config.ini", "w") as f:
                f.write(f"warp_res_init           = 3.2\n")
                f.write(f"warp_scaling            = 1 1 2 2 2 2\n")
                f.write(f"img_warp_space          = {templbrain}\n")
                f.write(f"lambda_reg              = 4.0e4 3.7e-1 3.1e-1 2.6e-1 2.2e-1 1.8e-1\n")
                f.write(f"hires                   = 0.5\n")
                f.write(f"optimiser_max_it_lowres = 10\n")
                f.write(f"optimiser_max_it_hires  = 5\n")
                f.write(f"\n")
                f.write(f"; Parameters relating to first scalar image pair\n")
                f.write(f"\n")
                f.write(f"img_ref_scalar      = {templbrain}\n")
                f.write(f"img_mov_scalar      = fast/T2_brain_tmp_ud.nii.gz\n")
                f.write(f"aff_ref_scalar      = {options.mmorfdir}/ident.mat\n")
                f.write(f"aff_mov_scalar      = T2_to_templ_linear.mat\n")
                f.write(f"use_implicit_mask   = 0\n")
                f.write(f"use_mask_ref_scalar = 1 1 1 1 1 1\n")
                f.write(f"use_mask_mov_scalar = 1 1 1 1 1 1\n")
                f.write(f"mask_ref_scalar     = {templmaskdil}\n")
                f.write(f"mask_mov_scalar     = T2_mask_dil\n")
                f.write(f"fwhm_ref_scalar     = 0.8 0.8 0.4 0.2 0.1 0.05\n")
                f.write(f"fwhm_mov_scalar     = 0.8 0.8 0.4 0.2 0.1 0.05\n")
                f.write(f"lambda_scalar       = 0.5 0.2 0.2 0.2 0.5 0.5\n")
                f.write(f"estimate_bias       = 1\n")
                f.write(f"bias_res_init       = 3.2\n")
                f.write(f"lambda_bias_reg     = 1e8 1e8 1e8 1e7 1e7 1e6\n")

            with open("mmorf/mmorf_config.ini") as f:
                LOG.debug(f.read())

            LOG.info(" - Running MMORF using config file (includes masks for template and T2 volume)")
            os.system(f"singularity run --nv {options.mmorfdir}/mmorf.sif --warp_out mmorf/T2_to_templ_warp --jac_det_out mmorf/T2_to_templ_jac --bias_out mmorf/T2_to_templ_bias --config mmorf/mmorf_config.ini |tee mmorf/mmorf.log")

            # # Apply the warp generated by mmorf
            fsl.applywarp("T2_to_templ_linear", ref=templbrain, out="mmorf/T2_to_templ_mmorf", warp="mmorf/T2_to_templ_warp")
            # ${FSLDIR}/bin/applywarp --ref=${tempBrain} --in=${anatdir}/T2_to_templ_linear.nii.gz --out=${mmorfdir}/T2_to_templ_mmorf --warp=${mmorfdir}/T2_to_templ_warp.nii.gz

            # # Check that this is working correctly
            fsl.slicer("mmorf/T2_to_templ_mmorf", templbrain, out="mmorf/mmorf.png")

        # # Clean and reorganize
        # # rm ${anatdir}/*tmp*
        makedirs("transforms", exist_ok=True)
        # # mv ${anatdir}/*templ* ${anatdir}/transforms
        os.system("mv *.mat transforms")

        LOG.info("Step 6: revise brain mask after non-linear registration to refine extraction")

        # Have only done this with ANTs - need to confirm whether ANTs or mmorf are better 
        if options.nonlinreg == "ants":
            LOG.info(" - Apply the inverse warp generated by ANTs to take brain mask to native space after linear registration, binary mask so use nearest neighbour interpolation")
            fsl.applywarp(templmask, ref="T2_reorient_ud", out="T2_mask", warp="ANTs/xfms/templ_to_T2_warp", interp="nn")
            # ${FSLDIR}/bin/applywarp --ref=${anatdir}/T2_reorient_ud --in=${mask} --out=${anatdir}/T2_mask --warp=${ANTsdir}/xfms/templ_to_T2_warp --interp=nn

            LOG.info(" - Use the mask created in the previous step to remove the skull from the T2 volume")
            fsl.fslmaths("T2_reorient_ud").mul("T2_mask").run("T2_brain_ud")
            # ${FSLDIR}/bin/fslmaths ${anatdir}/T2_reorient_ud -mul ${anatdir}/T2_mask ${anatdir}/T2_brain_ud

            # # Check that the brain masking was successful
            fsl.slicer("T2_reorient_ud", "T2_mask", out="skull_strip.png")

            LOG.info(" - Sanity check: Apply the warp generated by ANTs to take subject brain, after refined skull-stripping to template ")
            fsl.applywarp("T2_brain_ud", ref=templbrain, out="T2_brain_ud_to_templ", warp="ANTs/xfms/T2_to_templ_warp")
            # ${FSLDIR}/bin/applywarp --ref=${tempBrain} --in=${anatdir}/T2_brain_ud --out=${anatdir}/T2_brain_ud_to_templ --warp=${ANTsdir}/xfms/T2_to_templ_warp

            # # Check that this is working correctly
            fsl.slicer("T2_brain_ud_to_templ", templbrain, out="skull_strip_std_space.png")

        LOG.info("Step 7: segment brain into tissue classes in standard space")

        makedirs("seg/atlas", exist_ok=True)
        makedirs("seg/fast", exist_ok=True)

        # # Still need to decide whether FAST or ATLAS based segmentation is better

        LOG.info(" - ATLAS-BASED SEGMENTATION - apply non-linear warp generated by ANTs to take the probability maps to native space")
        fsl.applywarp(map_CSF, ref="T2_brain_ud", out="seg/atlas/T2_brain_ud_CSF", warp="ANTs/xfms/templ_to_T2_warp", interp="trilinear")
        fsl.applywarp(map_WM, ref="T2_brain_ud", out="seg/atlas/T2_brain_ud_WM", warp="ANTs/xfms/templ_to_T2_warp", interp="trilinear")
        fsl.applywarp(map_GM, ref="T2_brain_ud", out="seg/atlas/T2_brain_ud_GM", warp="ANTs/xfms/templ_to_T2_warp", interp="trilinear")
        # ${FSLDIR}/bin/applywarp --ref=${anatdir}/T2_brain_ud --in=${map_CSF} --out=${segAtlas}/T2_brain_ud_CSF --warp=${ANTsdir}/xfms/templ_to_T2_warp --interp=trilinear
        # ${FSLDIR}/bin/applywarp --ref=${anatdir}/T2_brain_ud --in=${map_WM} --out=${segAtlas}/T2_brain_ud_WM --warp=${ANTsdir}/xfms/templ_to_T2_warp --interp=trilinear
        # ${FSLDIR}/bin/applywarp --ref=${anatdir}/T2_brain_ud --in=${map_GM} --out=${segAtlas}/T2_brain_ud_GM --warp=${ANTsdir}/xfms/templ_to_T2_warp --interp=trilinear

        LOG.info(" - Binarise probability maps")
        fsl.fslmaths("seg/atlas/T2_brain_ud_CSF").thr(0.5).bin().run("seg/atlas/T2_brain_ud_CSF_bin")
        fsl.fslmaths("seg/atlas/T2_brain_ud_WM").thr(0.5).bin().run("seg/atlas/T2_brain_ud_WM_bin")
        fsl.fslmaths("seg/atlas/T2_brain_ud_GM").thr(0.5).bin().run("seg/atlas/T2_brain_ud_GM_bin")
        # $FSLDIR/bin/fslmaths ${segAtlas}/T2_brain_ud_CSF -thr 0.5 -bin ${segAtlas}/T2_brain_ud_CSF_bin
        # $FSLDIR/bin/fslmaths ${segAtlas}/T2_brain_ud_WM -thr 0.5 -bin ${segAtlas}/T2_brain_ud_WM_bin
        # $FSLDIR/bin/fslmaths ${segAtlas}/T2_brain_ud_GM -thr 0.5 -bin ${segAtlas}/T2_brain_ud_GM_bin

        LOG.info(" - FAST-BASED SEGMENTATION - segment brain into 3 tissue classes, in native space, using template probability maps as priors for the final segmentation as well (-P is a flag which doesn't require any inputs)")
        fsl.fast("T2_brain_ud", out="seg/fast/T2_brain_P", type=2, n_classes=3, a="transforms/templ_to_T2_linear.mat", A=[map_CSF, map_GM, map_WM], P=True)
        # ${FSLDIR}/bin/fast -t 2 -n 3 -a ${anatdir}/transforms/templ_to_T2_linear.mat -A ${map_CSF} ${map_GM} ${map_WM} -P -o ${segFast}/T2_brain_P ${anatdir}/T2_brain_ud

        LOG.info(" - Binarise PVE maps")
        fsl.fslmaths("seg/fast/T2_brain_P_pve_0").thr(0.5).bin().run("seg/fast/T2_brain_P_pve_0_bin")
        fsl.fslmaths("seg/fast/T2_brain_P_pve_1").thr(0.5).bin().run("seg/fast/T2_brain_P_pve_1_bin")
        fsl.fslmaths("seg/fast/T2_brain_P_pve_2").thr(0.5).bin().run("seg/fast/T2_brain_P_pve_2_bin")
        # $FSLDIR/bin/fslmaths ${segFast}/T2_brain_P_pve_0 -thr 0.5 -bin ${segFast}/T2_brain_P_pve_0_bin
        # $FSLDIR/bin/fslmaths ${segFast}/T2_brain_P_pve_1 -thr 0.5 -bin ${segFast}/T2_brain_P_pve_1_bin
        # $FSLDIR/bin/fslmaths ${segFast}/T2_brain_P_pve_2 -thr 0.5 -bin ${segFast}/T2_brain_P_pve_2_bin

        LOG.info("END: rodent_struct")
