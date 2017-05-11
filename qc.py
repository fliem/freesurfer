from nilearn import plotting, image
from nilearn import image as nlimage
from nilearn.plotting import plot_anat
import os
from collections import OrderedDict
import matplotlib.pylab as plt
import nibabel as nb
import numpy as np
# from niworkflows.viz.utils import cuts_from_bbox, robust_set_limits
from PIL import Image, ImageChops
from yattag import Doc

hemis = ["lh", "rh"]
hemis_full = {"lh": "left", "rh": "right"}


def plot_registration(anat_nii, fsid, out_dir, kind,
                      plot_params=None,
                      order=('z', 'x', 'y'), cuts=None,
                      estimate_brightness=False, label=None, contour=None, ext="pdf", subcort=False,
                      extra_out_label=""):
    """
    Adapted from https://github.com/poldracklab/niworkflows/blob/3d8b6de0bbd99ef4340535d565fb71c905b62ec5/niworkflows/viz/utils.py#L265
    Added directly save to file
    Plots the foreground and background views
    Default order is: axial, coronal, sagittal
    """

    plot_params = {} if plot_params is None else plot_params

    # Use default MNI cuts if none defined
    if cuts is None:
        raise NotImplementedError  # TODO

    out_files = []
    if estimate_brightness:
        plot_params = robust_set_limits(anat_nii.get_data().reshape(-1),
                                        plot_params)

    # FreeSurfer ribbon.mgz
    ribbon = contour is not None and \
             np.array_equal(np.unique(contour.get_data()),
                            [0, 2, 3, 41, 42])
    if ribbon:
        contour_data = contour.get_data() % 39
        white = nlimage.new_img_like(contour, contour_data == 2)
        pial = nlimage.new_img_like(contour, contour_data >= 2)

    # Plot each cut axis
    for i, mode in enumerate(list(order)):
        plot_params['display_mode'] = mode
        plot_params['cut_coords'] = cuts[mode]
        if i == 0:
            plot_params['title'] = label
        else:
            plot_params['title'] = None

        # Generate nilearn figure
        display = plot_anat(anat_nii, **plot_params)

        if ribbon:
            kwargs = {'levels': [0.5], 'linewidths': 0.5}
            display.add_contours(white, colors='b', **kwargs)
            display.add_contours(pial, colors='r', **kwargs)

        elif subcort:
            display = plotting.plot_roi(contour, anat_nii, cmap="prism", **plot_params)
            # subcort_data = contour.get_data()
            # cm = plt.cm.prism(np.unique(subcort_data))
            # kwargs = {'levels': [0.5], 'linewidths': .5}
            # for c, i in enumerate(np.unique(subcort_data)):
            #     subcort_sel = image.new_img_like(contour, subcort_data == i)
            #     display.add_contours(subcort_sel, colors=[cm[c]], **kwargs)

        elif contour is not None:
            display.add_contours(contour, levels=[.9])

        out_file = '{}_{}_{}{}.{}'.format(fsid, kind, mode, extra_out_label, ext)

        display.savefig(os.path.join(out_dir, out_file))
        display.close()

def robust_set_limits(data, plot_params):
    # copy of https://github.com/poldracklab/niworkflows/blob/3d8b6de0bbd99ef4340535d565fb71c905b62ec5/niworkflows/viz/utils.py#L265
    vmin = np.percentile(data, 15)
    if plot_params.get('vmin', None) is None:
        plot_params['vmin'] = vmin
    if plot_params.get('vmax', None) is None:
        plot_params['vmax'] = np.percentile(data[data > vmin], 99.8)

    return plot_params

def cuts_from_bbox(mask_nii, cuts=3):
    # copy of https://github.com/poldracklab/niworkflows/blob/3d8b6de0bbd99ef4340535d565fb71c905b62ec5/niworkflows
    # /viz/utils.py#L185
    """Finds equi-spaced cuts for presenting images"""
    from nibabel.affines import apply_affine
    mask_data = mask_nii.get_data()
    B = np.argwhere(mask_data > 0)
    start_coords = B.min(0)
    stop_coords = B.max(0) + 1

    vox_coords = []
    for start, stop in zip(start_coords, stop_coords):
        inc = abs(stop - start) / (cuts + 1)
        vox_coords.append([start + (i + 1) * inc for i in range(cuts)])

    ras_coords = []
    for cross in np.array(vox_coords).T:
        ras_coords.append(apply_affine(mask_nii.affine, cross).tolist())

    ras_cuts = [list(coords) for coords in np.transpose(ras_coords)]
    return {k: v for k, v in zip(['x', 'y', 'z'], ras_cuts)}



def remove_white_margins(image_file):
    # http://stackoverflow.com/questions/10615901/trim-whitespace-using-pil
    im = Image.open(image_file)
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        im = im.crop(bbox)
    im.save(image_file)


def export_parcellation(fs_dir, fsid, out_dir):
    for hemi in hemis:
        surf = os.path.join(fs_dir, fsid, "surf/{hemi}.inflated".format(hemi=hemi))
        annot = os.path.join(fs_dir, fsid, "label/{hemi}.aparc.annot".format(hemi=hemi))
        for view in ["lateral", "medial"]:
            output_file = os.path.join(out_dir, "{fsid}_parc_{hemi}_{view}.png".format(fsid=fsid, hemi=hemi,
                                                                                       view=view))
            d = plotting.plot_surf(surf, annot, cmap="Dark2", view=view, hemi=hemis_full[hemi], alpha=1,
                                   output_file=output_file, tight_layout=True)
            remove_white_margins(output_file)


# SURFACES
def export_surf(fs_dir, fsid, out_dir):
    brain = os.path.join(fs_dir, fsid, "mri/brain.mgz")
    brain_img = nb.load(brain)

    ribbon = os.path.join(fs_dir, fsid, "mri/ribbon.mgz")
    contour = nb.load(ribbon)

    cuts = cuts_from_bbox(contour, cuts=1)
    d = plot_registration(brain_img, fsid,
                          kind="surf",
                          out_dir=out_dir,
                          estimate_brightness=True,
                          order=["ortho"],
                          cuts={"ortho": (*cuts["x"], *cuts["y"], *cuts["z"])},
                          plot_params={"draw_cross": False},
                          contour=contour,
                          ext="svg")

    cuts = cuts_from_bbox(contour, cuts=10)
    d = plot_registration(brain_img, fsid,
                          kind="surf",
                          out_dir=out_dir,
                          estimate_brightness=True,
                          cuts=cuts,
                          contour=contour,
                          ext="svg")


def export_subcort(fs_dir, fsid, out_dir):
    brain = os.path.join(fs_dir, fsid, "mri/brain.mgz")
    brain_img = nb.load(brain)

    aseg_img = nb.load(os.path.join(fs_dir, fsid, "mri/aseg.mgz"))
    subcort_data = aseg_img.get_data() % 39
    subcort_data[subcort_data < 5] = 0
    subcort_img = image.new_img_like(aseg_img, subcort_data)

    cuts = cuts_from_bbox(subcort_img, cuts=1)
    d = plot_registration(brain_img, fsid,
                          kind="subcort",
                          out_dir=out_dir,
                          estimate_brightness=True,
                          order=["ortho"],
                          cuts={"ortho": (*cuts["x"], *cuts["y"], *cuts["z"])},
                          plot_params={"draw_cross": False},
                          contour=subcort_img,
                          subcort=True,
                          ext="svg")

    cuts = cuts_from_bbox(subcort_img, cuts=5)
    d = plot_registration(brain_img, fsid,
                          kind="subcort",
                          out_dir=out_dir,
                          estimate_brightness=True,
                          cuts=cuts,
                          plot_params={"draw_cross": False},
                          contour=subcort_img,
                          subcort=True,
                          ext="svg")




def create_subject_plots(fs_dir, fsid, out_base="00_qc/images"):
    out_dir = os.path.join(fs_dir, out_base, fsid)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    #
    export_subcort(fs_dir, fsid, out_dir)
    export_surf(fs_dir, fsid, out_dir)
    export_parcellation(fs_dir, fsid, out_dir)





#### REPORTS

def create_subject_report(qc_dir, fsid):
    out_dir = os.path.join(qc_dir, "reports")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    doc, tag, text = Doc().tagtext()

    images = OrderedDict()
    file_prefix = os.path.join("../images", fsid, fsid + "_")
    images["Cortical surfaces"] = {"horizontal": False, "files": ["surf_x.svg", "surf_y.svg", "surf_z.svg"]}
    images["Subcortical segmentation"] = {"horizontal": False, "files": ["subcort_x.svg", "subcort_y.svg",
                                                                         "subcort_z.svg"]}
    images["Cortical parcellation"] = {"horizontal": True, "files": ["parc_lh_lateral.png", "parc_lh_medial.png",
                                                                     "parc_rh_medial.png", "parc_rh_lateral.png"]}

    with tag('html'):
        with tag('body'):
            with tag("h1"):
                text("Freesurfer quality reports: %s" % fsid)

            for image_heading, info in images.items():
                with tag("h2"):
                    text(image_heading)
                with tag("div", klass="image123"):
                    if info["horizontal"]:
                        with tag("nobr"):
                            for image_file in info["files"]:
                                doc.stag("img", src=file_prefix + image_file)
                    else:
                        for image_file in info["files"]:
                            with tag("br"):
                                doc.stag("img", src=file_prefix + image_file)

    result = doc.getvalue()
    with open(os.path.join(out_dir, "%s_report.html" % fsid), "w") as fi:
        fi.write(result)


def create_group_report(qc_dir, fsid_list):
    image_list = ["surf_ortho.svg", "subcort_ortho.svg", "parc_lh_lateral.png", "parc_lh_medial.png",
                  "parc_rh_medial.png", "parc_rh_lateral.png"]
    fsid_list = sorted(fsid_list)

    doc, tag, text = Doc().tagtext()
    with tag('html'):
        with tag('body'):
            with tag("h1"):
                text("Freesurfer quality reports. Group overview.")
                for fsid in fsid_list:
                    subject_report_file = "reports/{}_report.html".format(fsid)
                    with tag("a", href=subject_report_file, target="_blank"):
                        with tag("h2"):
                            text(fsid)
                        with tag("nobr"):
                            for images_temp in image_list:
                                image = os.path.join("images", fsid, fsid + "_" + images_temp)
                                doc.stag("img", src=image)

    result = doc.getvalue()
    with open(os.path.join(qc_dir, "freesurfer_qc_group_report.html"), "w") as fi:
        fi.write(result)



if __name__ == "__main__":
    ###

    fs_dir = "/Users/franzliem/Desktop/ds114_test1_freesurfer_precomp_v6.0.0"
    qc_dir = "/Users/franzliem/Desktop/ds114_test1_freesurfer_precomp_v6.0.0/00_qc"

    fsid = "sub-01"
    create_subject_plots(fs_dir, fsid)
    create_subject_report(qc_dir, fsid)

    fsid = "sub-02"
    create_subject_plots(fs_dir, fsid)
    create_subject_report(qc_dir, fsid)

    create_group_report(qc_dir, ["sub-01", "sub-02"])
