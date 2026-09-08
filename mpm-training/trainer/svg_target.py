"""SVG targets compiled once to Cairo vector recordings, rotated before sampling."""
from functools import lru_cache
import sys
import xml.etree.ElementTree as ET

import numpy as np


def _no_external_resources(url, resource_type):
    raise ValueError('SVG targets must be self-contained (no external resources)')


@lru_cache(maxsize=8)
def _recording(source):
    import cairocffi as cairo
    from cairosvg.parser import Tree
    from cairosvg.surface import Surface
    from defusedxml.ElementTree import fromstring

    try:
        root = fromstring(source)
    except ET.ParseError as error:
        raise ValueError(f"invalid SVG XML: {error}") from error
    if root.tag != '{http://www.w3.org/2000/svg}svg':
        raise ValueError('target must be an SVG document with the SVG namespace')
    viewbox = root.get('viewBox')
    if viewbox is None:
        try:
            width = float(root.get('width', '').removesuffix('px'))
            height = float(root.get('height', '').removesuffix('px'))
        except ValueError as error:
            raise ValueError('SVG requires a viewBox or numeric width and height') from error
        viewbox = f'0 0 {width} {height}'
    box = np.asarray([float(v) for v in viewbox.replace(',', ' ').split()])
    if box.shape != (4,) or not np.isfinite(box).all() or min(box[2:]) <= 0:
        raise ValueError('invalid SVG viewBox')
    root.set('viewBox', viewbox)
    root.set('width', '1'); root.set('height', '1')
    # Keep aspect ratio even for non-square artboards.
    root.set('preserveAspectRatio', 'xMidYMid meet')

    class RecordingSurface(Surface):
        device_units_per_user_units = 1

        def _create_surface(self, width, height):
            return cairo.RecordingSurface(cairo.CONTENT_COLOR_ALPHA, None), width, height

    tree = Tree(bytestring=ET.tostring(root), url_fetcher=_no_external_resources)
    surface = RecordingSurface(tree, None, 96, output_width=1, output_height=1)
    return surface.cairo


def render_svg(source, resolution, angle=0., transform=None, supersample=2):
    """Return alpha and premultiplied RGB in simulation (y-up) coordinates.

    Angles are clockwise, matching the trainer's row-vector convention.
    Cairo replays vector commands under the transform; no bitmap is rotated.
    """
    import cairocffi as cairo
    n = resolution * supersample
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, n, n)
    context = cairo.Context(surface)
    context.scale(n, n)
    if transform is not None:
        cx, cy, scale = transform
        context.translate(.5, .5)
        context.rotate(float(angle))
        context.scale(scale, scale)
        context.translate(-cx, -cy)
    context.set_source_surface(_recording(source))
    context.paint()
    surface.flush()
    raw = np.ndarray((n, n, 4), dtype=np.uint8, buffer=surface.get_data(),
                     strides=(surface.get_stride(), 4, 1))
    channels = [2, 1, 0, 3] if sys.byteorder == 'little' else [1, 2, 3, 0]
    rgba = raw[::-1, :, channels].astype(np.float64) / 255
    rgba = rgba.reshape(resolution, supersample, resolution, supersample, 4).mean(axis=(1, 3))
    return rgba[..., 3], rgba[..., :3]


def svg_transform(source):
    # Estimate the alpha centroid once, then fit every occupied point inside a
    # radius of .48 so no orientation can crop the target or change its mass.
    mask, _ = render_svg(source, 512)
    ys, xs = np.nonzero(mask > 0)
    mass = float(mask.sum())
    if mass <= 0:
        raise ValueError('SVG target has no visible artwork')
    weights = mask[ys, xs]
    cx = float(((xs+.5)*weights).sum()/mass/512)
    cy = float(((ys+.5)*weights).sum()/mass/512)
    radius = np.hypot((xs+.5)/512-cx, (ys+.5)/512-cy).max() + np.sqrt(2)/512
    return (cx, 1-cy, float(.48/radius))
