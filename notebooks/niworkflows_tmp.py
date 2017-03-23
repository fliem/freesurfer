#https://github.com/poldracklab/niworkflows/blob/3d8b6de0bbd99ef4340535d565fb71c905b62ec5/niworkflows/viz/utils.py#L265
import numpy as np
from nilearn import plotting, image



def cuts_from_bbox(mask_nii, cuts=3):
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

def robust_set_limits(data, plot_params):
    vmin = np.percentile(data, 15)
    if plot_params.get('vmin', None) is None:
        plot_params['vmin'] = vmin
    if plot_params.get('vmax', None) is None:
        plot_params['vmax'] = np.percentile(data[data > vmin], 99.8)

    return plot_params

def plot_registration(anat_nii, div_id, plot_params=None,
                      order=('z', 'x', 'y'), cuts=None,
                      estimate_brightness=False, label=None, contour=None,
                      compress='auto'):
    """
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
        white = image.new_img_like(contour, contour_data == 2)
        pial = image.new_img_like(contour, contour_data >= 2)

    # Plot each cut axis
    for i, mode in enumerate(list(order)):
        out_file = '{}_{}.svg'.format(div_id, mode)
        plot_params['display_mode'] = mode
        plot_params['cut_coords'] = cuts[mode]
        if i == 0:
            plot_params['title'] = label
        else:
            plot_params['title'] = None

        # Generate nilearn figure
        display = plotting.plot_anat(anat_nii, **plot_params)
        if ribbon:
            kwargs = {'levels': [0.5], 'linewidths': 0.5}
            display.add_contours(white, colors='b', **kwargs)
            display.add_contours(pial, colors='r', **kwargs)
        elif contour is not None:
            display.add_contours(contour, levels=[.9])
#        svg = extract_svg(display, compress=compress)
#        display.close()

        # Find and replace the figure_1 id.

#        try:
#            xml_data = etree.fromstring(svg)
#        except etree.XMLSyntaxError as e:
#            NIWORKFLOWS_LOG.info(e)
#            return
#        find_text = etree.ETXPath("//{%s}g[@id='figure_1']" % (SVGNS))
#        find_text(xml_data)[0].set('id', '%s-%s-%s' % (div_id, mode, uuid4()))
#
#        out_files.append(etree.tostring(xml_data))
#
#    return out_files
