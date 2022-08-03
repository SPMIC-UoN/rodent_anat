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

from fsl.data.image import Image
import fsl.wrappers as fsl
from fsl.utils.imcp import immv, imcp

from .reorient import to_std_orientation
from .utils import working_dir, makedirs

LOG = logging.getLogger(__name__)

def run(options):
    LOG.info("START: rodent_anat")
    LOG.info("Step 1: housekeeping, error checking and reorientation")

    outdir = os.path.abspath(options.output)

    # Templates FIXME need better system for customization
    templdir=options.template
    templ_mask = os.path.join(templdir, "SIGMA_ExVivo_Brain_Mask_No_OlfBulb.nii.gz")
    templ_mask_dil = os.path.join(templdir, "SIGMA_ExVivo_Brain_Mask_No_OlfBulb.nii.gz")
    #templ_mask_dil = os.path.join(templdir, "SIGMA_ExVivo_Brain_Mask_No_OlfBulb_dil.nii.gz")
    templ_brain = os.path.join(templdir, "SIGMA_ExVivo_Brain_Template_Masked_No_OlfBulb.nii.gz")
    #templ_brain = os.path.join(templdir, "SIGMA_ExVivo_Brain_Template.nii.gz")
    templ_gm = os.path.join(templdir, "SIGMA_ExVivo_GM_No_OlfBulb.nii.gz")
    templ_wm = os.path.join(templdir, "SIGMA_ExVivo_WM_No_OlfBulb.nii.gz")
    templ_csf = os.path.join(templdir, "SIGMA_ExVivo_CSF_No_OlfBulb.nii.gz")

    # Check that T2 is an image
    T2 = Image(options.input)
    
    with working_dir(outdir):
        if options.noreorient:
            LOG.info(" - Copying T2 to output directory")
            T2.save("T2_reorient.nii.gz")
        else:
            to_std_orientation(T2.dataSource, "T2_reorient.nii.gz")

        LOG.info("Step 2: Performing initial brain extraction and registration to standard template")

        # Files generated
        # T2_templ_linear - T2 registered to template
        # T2_brain_templ_linear - Brain extracted T2 aligned with template
        # T2_to_templ_linear.mat - linear transform
        # templ_to_T2_linear.mat - linear transform
        # T2_brain_linear - Brain extracted T2
        # T2_brain_mask_linear  - template mask aligned to T2 image

        LOG.info(" - Performing linear registration of T2 to brain-only template")
        fsl.flirt("T2_reorient", templ_brain, out="T2_templ_linear", omat="T2_to_templ_linear.mat") # -dof 12

        LOG.info(" - Inverting linear registration of T2 to brain-only template")
        fsl.invxfm("T2_to_templ_linear.mat", "templ_to_T2_linear.mat")

        LOG.info(" - Transforming dilated template brain mask to T2 native space")
        fsl.applyxfm(templ_mask_dil, "T2_reorient", "templ_to_T2_linear.mat", interp="nearestneighbour", out="T2_brain_mask_linear")

        LOG.info(" - Removing skull from T2")
        fsl.fslmaths("T2_reorient").mul("T2_brain_mask_linear").run("T2_brain_linear")

        LOG.info(" - Transforming brain extracted T2 into standard template space")
        fsl.applyxfm("T2_brain_linear", templ_brain, "T2_to_templ_linear.mat", out="T2_brain_templ_linear", interp="trilinear")

        if not options.nobias:
            LOG.info("Step 3: correct bias-field in skull-stripped T2 volume")
            imcp("T2_reorient", "T2_reorient_nobiascorr")
            imcp("T2_brain_linear", "T2_brain_linear_nobiascorr")
            imcp("T2_brain_mask_linear", "T2_brain_mask_linear_nobiascorr")
            shutil.copyfile("T2_to_templ_linear.mat", "T2_to_templ_linear_nobiascorr.mat")
            shutil.copyfile("templ_to_T2_linear.mat", "templ_to_T2_linear_nobiascorr.mat")
            imcp("T2_brain_templ_linear", "T2_brain_templ_linear_nobiascorr")
            imcp("T2_templ_linear", "T2_templ_linear_nobiascorr")
            
            # Files generated
            # T2_brain_linear - Brain extracted T2 without bias field
            # T2 - T2 without bias field

            LOG.info(f" - Running FAST to estimate bias field: smoothing (FWHM) = {options.biassmooth}mm")
            makedirs("fast", exist_ok=True)
            fsl.fast("T2_brain_linear", out="fast/T2_brain", b=True, B=True, nopve=True, lowpass=options.biassmooth, type=2, segments=False)

            # # Clean up directory
            # rm ${fastdir}/T2_brain_tmp_seg*

            LOG.info(" - Using estimated bias field to correct T2")
            imcp("fast/T2_brain_restore", "T2_brain_linear", overwrite=True)
            fsl.fslmaths("T2_reorient").div("fast/T2_brain_bias").run("T2_reorient")

            LOG.info("Step 4: Refine the linear registration using the bias corrected brain")

            # Files generated
            # T2_brain_templ_linear - undistorted brain-extracted T2 registered to template
            # T2_to_templ_linear.mat - linear transform
            # templ_to_T2_linear.mat - linear transform
            # T2_brain_mask_linear - Template brain mask aligned with undistorted T2
            # T2_templ_linear - Undistorted T2 aligned with template

            LOG.info(" - Register the bias-corrected brain to standard brain-only template")
            fsl.flirt("T2_brain_linear", templ_brain, out="T2_brain_templ_linear", omat="T2_to_templ_linear.mat")

            LOG.info(" - Inverting linear transform of bias-corrected brain to standard brain-only template")
            fsl.invxfm("T2_to_templ_linear.mat", "templ_to_T2_linear.mat")

            LOG.info(" - Transforming dilated template brain mask to bias-corrected T2 native space")
            fsl.applyxfm(templ_mask_dil, "T2_brain_linear", "templ_to_T2_linear.mat", interp="nearestneighbour", out="T2_brain_mask_linear")

            LOG.info(" - Transforming bias-corrected T2 into standard template space")
            fsl.applyxfm("T2_reorient", templ_brain, "T2_to_templ_linear.mat", out="T2_templ_linear", interp="trilinear")

            #LOG.info(" - Comparing success of linear registration on skull-stripped data with and without bias correction")
            #fsl.flirt("T2_brain_dil", templ_brain, out="T2_brain_templ_linear_biased", dof=12)
        else:
            LOG.info("Bias field correction (Step 3, 4) disabled")

        LOG.info("Step 5: Perform nonlinear registration to map T2 to the standard template")

        if options.nonlinreg == "ants":
            LOG.info(" - Running nonlinear registration using ANTs")
            makedirs("ants", exist_ok=True)

            # Run the script antsRegistrationSyN.sh using brain-extracted bias-corrected scan and brain-only template, default is double precision, 
            # use histogram matching (-j) because template and scan are the from the same modality, use mask (-x) for template so that exclude the
            # olfactory bulb, have chosen dilated version of mask
            # Note: antsRegistrationSyN actually includes optimisation of rigid, affine and SyN transformations
            # Note: antsRegistrationSyN works best on skull-stripped, bias corrected data
            # see: https://github.com/ANTsX/ANTs/wiki/Anatomy-of-an-antsRegistration-call
            os.system(f"{options.antspath}/antsRegistrationSyN.sh -d 3 -f {templ_brain} -m T2_brain_templ_linear.nii.gz -j 1 -o ants/T2_to_templ_ -x {templ_mask_dil} |tee ants/ants.log")

            LOG.info(" - Converting ANTs transforms so that they are compatible with FSL (ITK transform (RAS) matrix to FSL xfm)")
            os.system(f"{options.c3dpath}/c3d_affine_tool -ref {templ_brain} -src T2_brain_templ_linear.nii.gz -itk ants/T2_to_templ_ants_0GenericAffine.mat -ras2fsl -o ants/T2_to_templ_linear.mat")

            LOG.info(" - Performing multicomponent split (-mcs)")
            os.system(f"{options.c3dpath}/c3d -mcs ants/T2_to_templ_1Warp.nii.gz -oo ants/wx.nii.gz ants/wy.nii.gz ants/wz.nii.gz")

            LOG.info(" - Flipping warp")
            fsl.fslmaths("ants/wy").mul(-1).run("ants/i_wy")

            LOG.info(" - Concatenating components to give whole warp")
            # FIXME
            os.system("fslmerge -t ants/T2_to_templ_warp ants/wx ants/i_wy ants/wz")

            makedirs("ants/xfms", exist_ok=True)
                    
            LOG.info(" - Combining linear transform generated by ANTs with linear transform generated in step 4")
            fsl.concatxfm("ants/T2_to_templ_linear.mat", "T2_to_templ_linear.mat", "ants/T2_to_templ_linear_x2.mat")
            # ${FSLDIR}/bin/convert_xfm -omat ${antsdir}/T2_to_templ_linear_x2.mat -concat ${antsdir}/ants_T2_to_templ_affine_flirt.mat ${anatdir}/T2_to_templ_linear.mat

            LOG.info(" - Combining linear transform generated in previous step with the warp generated by ANTs")
            fsl.convertwarp(ref=templ_brain, premat="ants/T2_to_templ_linear_x2.mat", warp1="ants/T2_to_templ_warp", out="ants/T2_to_templ_warp_combined")
            # ${FSLDIR}/bin/convertwarp --ref=${tempBrain} --premat=${antsdir}/T2_to_templ_linear_x2.mat --warp1=${antsdir}/ants_T2_to_templ_warp_fnirt --out=${antsdir}/xfms/T2_to_templ_warp
            # # ${FSLDIR}/bin/convertwarp --ref=${tempBrain} --premat=${antsdir}/ants_T2_to_templ_affine_flirt.mat --warp1=${antsdir}/ants_T2_to_templ_warp_fnirt --out=${antsdir}/xfms/T2_to_templ_warp

            LOG.info(" - Generating inverse of warp")
            fsl.invwarp("ants/xfms/T2_to_templ_warp_combined", "T2_brain_templ_linear", out="ants/templ_to_T2_warp_combined")
            
            T2_to_templ_warp = "ants/T2_to_templ_warp_combined"
            templ_to_T2_warp = "ants/templ_to_T2_warp_combined"

        elif options.nonlinreg == "mmorf":
            LOG.info(" - Running nonlinear registration using MMORF")

            LOG.info(" - Creating MMORF config file")
            makedirs("mmorf", exist_ok=True)
            with open("mmorf/mmorf_config.ini", "w") as f:
                f.write(f"warp_res_init           = 3.2\n")
                f.write(f"warp_scaling            = 1 1 2 2 2 2\n")
                f.write(f"img_warp_space          = {templ_brain}\n")
                f.write(f"lambda_reg              = 4.0e4 3.7e-1 3.1e-1 2.6e-1 2.2e-1 1.8e-1\n")
                f.write(f"hires                   = 0.5\n")
                f.write(f"optimiser_max_it_lowres = 10\n")
                f.write(f"optimiser_max_it_hires  = 5\n")
                f.write(f"\n")
                f.write(f"; Parameters relating to first scalar image pair\n")
                f.write(f"\n")
                f.write(f"img_ref_scalar      = {templ_brain}\n")
                f.write(f"img_mov_scalar      = T2_brain_linear.nii.gz\n")
                f.write(f"aff_ref_scalar      = {options.mmorfdir}/ident.mat\n")
                f.write(f"aff_mov_scalar      = T2_to_templ_linear.mat\n")
                f.write(f"use_implicit_mask   = 0\n")
                f.write(f"use_mask_ref_scalar = 1 1 1 1 1 1\n")
                f.write(f"use_mask_mov_scalar = 1 1 1 1 1 1\n")
                f.write(f"mask_ref_scalar     = {templ_mask_dil}\n")
                f.write(f"mask_mov_scalar     = T2_brain_mask_linear\n")
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
            fsl.applywarp("T2_templ_linear", ref=templ_brain, out="mmorf/T2_templ_mmorf", warp="mmorf/T2_to_templ_warp")
            # ${FSLDIR}/bin/applywarp --ref=${tempBrain} --in=${anatdir}/T2_to_templ_linear.nii.gz --out=${mmorfdir}/T2_to_templ_mmorf --warp=${mmorfdir}/T2_to_templ_warp.nii.gz

            LOG.info(" - Combining linear transform generated in previous step with the warp generated by MMORF")
            fsl.convertwarp(ref=templ_brain, premat="T2_to_templ_linear.mat", warp1="mmorf/T2_to_templ_warp", out="mmorf/T2_to_templ_warp_combined")
            
            LOG.info(" - Generating inverse of warp")
            fsl.invwarp("mmorf/T2_to_templ_warp_combined", ref="T2_templ_linear", out="mmorf/templ_to_T2_warp_combined")

            T2_to_templ_warp = "mmorf/T2_to_templ_warp_combined"
            templ_to_T2_warp = "mmorf/templ_to_T2_warp_combined"

        # # Clean and reorganize
        #makedirs("transforms", exist_ok=True)
        # # mv ${anatdir}/*templ* ${anatdir}/transforms
        #os.system("mv *.mat transforms")

        LOG.info("Step 6: revise brain mask after non-linear registration to refine extraction")

        # Files generated
        # T2_brain_mask - Template brain mask aligned with T2
        # T2_brain - Brain-extracted T2
        # T2_brain_templ - Brain-extracted T2 aligned to template
        # T2_templ - T2 aligned with template
        
        imcp("T2_brain", "T2_brain_linear")
        imcp("T2_brain_mask", "T2_brain_mask_linear")

        LOG.info(" - Apply the inverse warp to take brain mask to native space after linear registration, binary mask so use nearest neighbour interpolation")
        fsl.applywarp(templ_mask, ref="T2_reorient", out="T2_brain_mask", warp=templ_to_T2_warp, interp="nn")

        LOG.info(" - Use the mask created in the previous step to remove the skull from the T2 volume")
        fsl.fslmaths("T2_reorient").mul("T2_brain_mask").run("T2_brain")

        LOG.info(" - Apply the warp to take subject brain, after refined skull-stripping to template ")
        fsl.applywarp("T2_reorient", ref=templ_brain, out="T2_templ", warp=T2_to_templ_warp)

        LOG.info(" - Apply the warp to take subject brain, after refined skull-stripping to template ")
        fsl.applywarp("T2_brain", ref=templ_brain, out="T2_brain_templ", warp=T2_to_templ_warp)

        LOG.info("Step 7: segment brain into tissue classes in standard space")

        makedirs("seg/atlas", exist_ok=True)
        makedirs("seg/fast", exist_ok=True)

        # # Still need to decide whether FAST or ATLAS based segmentation is better

        LOG.info(" - ATLAS-BASED SEGMENTATION - apply non-linear warp to take the probability maps to native space")
        fsl.applywarp(templ_csf, ref="T2_brain", out="seg/atlas/T2_brain_CSF", warp=templ_to_T2_warp, interp="trilinear")
        fsl.applywarp(templ_wm, ref="T2_brain", out="seg/atlas/T2_brain_WM", warp=templ_to_T2_warp, interp="trilinear")
        fsl.applywarp(templ_gm, ref="T2_brain", out="seg/atlas/T2_brain_GM", warp=templ_to_T2_warp, interp="trilinear")

        LOG.info(" - Binarise probability maps")
        fsl.fslmaths("seg/atlas/T2_brain_CSF").thr(0.5).bin().run("seg/atlas/T2_brain_CSF_bin")
        fsl.fslmaths("seg/atlas/T2_brain_WM").thr(0.5).bin().run("seg/atlas/T2_brain_WM_bin")
        fsl.fslmaths("seg/atlas/T2_brain_GM").thr(0.5).bin().run("seg/atlas/T2_brain_GM_bin")

        LOG.info(" - FAST-BASED SEGMENTATION - segment brain into 3 tissue classes, in native space, using template probability maps as priors for the final segmentation as well (-P is a flag which doesn't require any inputs)")
        fsl.fast("T2_brain", out="seg/fast/T2_brain_P", type=2, n_classes=3, a="templ_to_T2_linear.mat", A=(templ_csf, templ_gm, templ_wm), P=True)

        LOG.info(" - Binarise PVE maps")
        fsl.fslmaths("seg/fast/T2_brain_P_pve_0").thr(0.5).bin().run("seg/fast/T2_brain_P_pve_0_bin")
        fsl.fslmaths("seg/fast/T2_brain_P_pve_1").thr(0.5).bin().run("seg/fast/T2_brain_P_pve_1_bin")
        fsl.fslmaths("seg/fast/T2_brain_P_pve_2").thr(0.5).bin().run("seg/fast/T2_brain_P_pve_2_bin")

        LOG.info("END: rodent_struct")
