#https://github.com/poldracklab/niworkflows/blob/3d8b6de0bbd99ef4340535d565fb71c905b62ec5/niworkflows/viz/utils.py#L265
import numpy as np
from nilearn import plotting, image
import shutil, re


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

def svg2str(display_object, dpi=300):
    """
    Serializes a nilearn display object as a string
    """
    from io import StringIO
    image_buf = StringIO()
    #     # FL with surfs: ---> 42     display_object.frame_axes.figure.savefig(
    #     43         image_buf, dpi=dpi, format='svg',
    #      44         facecolor='k', edgecolor='k')
    #
    # AttributeError: 'Figure' object has no attribute 'frame_axes'
    try:
        display_object.frame_axes.figure.savefig(
            image_buf, dpi=dpi, format='svg',
            facecolor='k', edgecolor='k')
    except:
        display_object.savefig(
                image_buf, dpi=dpi, format='svg',
                facecolor='k', edgecolor='k')
    return image_buf.getvalue()

def extract_svg(display_object, dpi=300, compress='auto'):
    """
    Removes the preamble of the svg files generated with nilearn
    """
    image_svg = svg2str(display_object, dpi)
    if compress == True or compress == 'auto':
        image_svg = svg_compress(image_svg, compress)
    image_svg = re.sub(' height="[0-9]+[a-z]*"', '', image_svg, count=1)
    image_svg = re.sub(' width="[0-9]+[a-z]*"', '', image_svg, count=1)
    image_svg = re.sub(' viewBox',
                       ' preseveAspectRation="xMidYMid meet" viewBox',
                       image_svg, count=1)
    start_tag = '<svg '
    start_idx = image_svg.find(start_tag)
    end_tag = '</svg>'
    end_idx = image_svg.rfind(end_tag)
    if start_idx is -1 or end_idx is -1:
        NIWORKFLOWS_LOG.info('svg tags not found in extract_svg')
    # rfind gives the start index of the substr. We want this substr
    # included in our return value so we add its length to the index.
    end_idx += len(end_tag)
    return image_svg[start_idx:end_idx]


def svg_compress(image, compress='auto'):
    ''' takes an image as created by nilearn.plotting and returns a blob svg.
    Performs compression (can be disabled). A bit hacky. '''

    # Compress the SVG file using SVGO
    if (shutil.which("svgo") and compress == 'auto') or compress is True:

        p = subprocess.run("svgo -i - -o - -q -p 3 --pretty --disable=cleanupNumericValues",
                           input=image.encode('utf-8'), stdout=subprocess.PIPE,
                           shell=True, check=True)
        image = p.stdout.decode('utf-8')

    # Convert all of the rasters inside the SVG file with 80% compressed WEBP
    if (shutil.which("cwebp") and compress == 'auto') or compress == True:
        new_lines = []
        with StringIO(image) as fp:
            for line in fp:
                if "image/png" in line:
                    tmp_lines = [line]
                    while "/>" not in line:
                        line = fp.readline()
                        tmp_lines.append(line)
                    content = ''.join(tmp_lines).replace('\n', '').replace(
                        ',  ', ',')

                    left = content.split('base64,')[0] + 'base64,'
                    left = left.replace("image/png", "image/webp")
                    right = content.split('base64,')[1]
                    png_b64 = right.split('"')[0]
                    right = '"' + '"'.join(right.split('"')[1:])

                    p = subprocess.run("cwebp -quiet -noalpha -q 80 -o - -- -",
                                       input=base64.b64decode(png_b64),
                                       stdout=subprocess.PIPE,
                                       shell=True, check=True)
                    webpimg = base64.b64encode(p.stdout).decode('utf-8')
                    new_lines.append(left + webpimg + right)
                else:
                    new_lines.append(line)
        lines = new_lines
    else:
        lines = image.splitlines()

    svg_start = 0
    for i, line in enumerate(lines):
        if '<svg ' in line:
            svg_start = i
            continue

    image_svg = lines[svg_start:]  # strip out extra DOCTYPE, etc headers
    return ''.join(image_svg)  # straight up giant string


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
    displays = []
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
        displays.append(display)
    return displays
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
