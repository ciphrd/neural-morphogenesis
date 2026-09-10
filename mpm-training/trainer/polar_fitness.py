"""IsoNCA image loss: sharpen, polar sample, minimize squared error over O(2).

Reference: google-research/self-organising-systems/isotropic_nca/
blogpost_isonca_single_seed_pytorch.ipynb (InvariantLoss and sharpen_filter).
We use an exactly periodic angular lattice. Radius covers the entire Cartesian
canvas, not just its inscribed disk, so corner details cannot disappear. There
is no radial Jacobian weighting. Channel errors sum angles and average radii;
the training objective normalizes by alpha energy and prioritizes shape. The fixed sampling grid depends only on the target and resolution.
"""
from config import CONFIG
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from domain_fitness import (DomainEvaluation, MatchMetrics, centered_triangles,
                            rasterize_triangles)
from raster import RasterFitnessBreakdown


@dataclass
class PolarDiagnostics:
    target: np.ndarray
    candidate: np.ndarray
    aligned_target: np.ndarray
    losses: np.ndarray  # [unreflected/reflected, angular shift]
    shift: int
    reflected: bool
    radius: float
    objective: dict | None = None

    @property
    def angle(self):
        return 2*np.pi*self.shift/self.target.shape[1]


# Ignore only clipping roundoff at exact triangle/pixel boundaries. This is
# geometric support, never a threshold on RGB or luminance.
COVERAGE_EPSILON = 1e-10


def opaque_image(coverage, premultiplied_rgb=None):
    """Identical loss-image preparation for triangle and target rasters.

    Any covered pixel is opaque, including black and semitransparent source
    material. Divide by coverage to recover straight RGB before setting alpha
    to one. Empty pixels have zero RGB and alpha. Overlaps average their RGB.
    """
    coverage = np.asarray(coverage, dtype=float)
    occupied = coverage > COVERAGE_EPSILON
    alpha = occupied.astype(float)
    if premultiplied_rgb is None:
        return alpha[..., None]
    rgb = np.zeros((*coverage.shape, 3), dtype=float)
    np.divide(premultiplied_rgb, coverage[..., None], out=rgb,
              where=occupied[..., None])
    return np.concatenate([np.clip(rgb, 0, 1), alpha[..., None]], axis=-1)


def sharpen(image):
    # torchvision's 5x5, sigma=1 blur uses reflect padding (scipy 'mirror').
    return image + 2*(image-gaussian_filter(image, sigma=(1, 1, 0), radius=(2, 2, 0), mode='mirror'))


def polar_sample(image, center, radius, radial_samples, angular_samples):
    """HWC image, pixel-index center (x,y); rows are increasing simulation y."""
    radii = np.linspace(.25, radius, radial_samples)[:, None]
    angles = np.arange(angular_samples)*2*np.pi/angular_samples
    coords = np.array([center[1]+radii*np.sin(angles), center[0]+radii*np.cos(angles)])
    return np.stack([map_coordinates(image[..., c], coords, order=1,
                                    mode='grid-constant', cval=0., prefilter=False)
                     for c in range(image.shape[2])], axis=-1)


def shape_priority_terms(channel_errors, shape_energy, color_weight=1.0):
    """Return normalized shape loss and bounded, continuously gated color loss.

    The derivative with respect to shape error stays positive: reducing the
    color gate by making the shape worse can never improve this objective.
    """
    config = CONFIG['polarFitness']
    cap, floor, scale = (float(config[k]) for k in
                         ('maxColorContribution', 'colorFloor', 'shapeScale'))
    if not (np.isfinite([cap, floor, scale, color_weight, shape_energy]).all()
            and 0 <= cap < 1 and 0 < floor <= 1 and scale > 0
            and cap*(1-floor) < scale and 0 <= color_weight <= 1 and shape_energy > 0):
        raise ValueError('invalid shape-priority weights: require cap*(1-floor) < shapeScale and color weight in [0,1]')
    errors = np.maximum(np.asarray(channel_errors, float), 0.) / shape_energy
    shape = errors[..., -1]
    if errors.shape[-1] == 1:
        return shape, np.zeros_like(shape)
    color = errors[..., :-1].mean(axis=-1)
    gate = floor + (1-floor)/(1+shape/scale)
    contribution = cap*color_weight*gate*(color/(1+color))
    return shape, contribution


def match_polar(candidate, target, target_fft=None, *, shape_priority=False, color_weight=1.0):
    """Return loss curves and selected reference. Checkable by direct np.roll SSE.

    Reflection is theta -> -theta (index zero stays fixed); a shift k then
    rotates that reference CCW by 2*pi*k/N in simulation coordinates.
    """
    candidate = np.asarray(candidate, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if candidate.shape != target.shape or candidate.ndim != 3 or not np.isfinite(candidate).all() or not np.isfinite(target).all():
        raise ValueError('polar inputs must be finite, matching radius-angle-channel arrays')
    n = target.shape[1]
    x = np.fft.rfft(candidate, axis=1)
    y = np.fft.rfft(target, axis=1) if target_fft is None else target_fft
    norms = np.square(candidate).sum(axis=1) + np.square(target).sum(axis=1)
    correlations = [np.fft.irfft(x*np.conj(y), n=n, axis=1),
                    np.fft.irfft(x*y, n=n, axis=1)]
    channel_errors = np.maximum(np.stack([
        (norms[:, None, :]-2*c).mean(axis=0) for c in correlations]), 0.)
    if shape_priority:
        energy = max(float(np.square(target[..., -1]).sum(axis=1).mean()), 1e-12)
        shape, color = shape_priority_terms(channel_errors, energy, color_weight)
        losses = shape + color
    else:
        losses = channel_errors.mean(axis=-1)
    reflection, shift = np.unravel_index(np.argmin(losses), losses.shape)
    reference = target[:, (-np.arange(n)) % n] if reflection else target
    aligned = np.roll(reference, int(shift), axis=1)
    return losses, int(shift), bool(reflection), aligned


class PolarTarget:
    def __init__(self, target, mask, use_color):
        n = len(mask)
        self.n = n
        self.center = np.asarray(target.center)
        # Fixed support includes all four target-canvas corners + blur support.
        corners = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.radius = float(np.max(np.linalg.norm(corners-self.center, axis=1)) + 3/n)
        self.pad = int(np.ceil(self.radius*n))+3
        self.size = n+2*self.pad
        self.center_pixels = self.center*n + self.pad-.5
        self.radial_samples = int(np.ceil(self.radius*n))
        self.angular_samples = int(np.ceil(2*np.pi*self.radius*n))
        rgb = target.color_raster(n) if use_color else None
        self.channels = 1 if rgb is None else 4
        image = opaque_image(mask, rgb)
        self.mass = float(image[..., -1].sum())
        self.image = np.pad(image, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)))
        self.polar = self.sample(sharpen(self.image))
        self.fft = np.fft.rfft(self.polar, axis=1)

    def sample(self, image):
        return polar_sample(image, self.center_pixels, self.radius*self.n,
                            self.radial_samples, self.angular_samples)

    def posed_image(self, shift, reflected):
        y, x = np.indices((self.size, self.size), dtype=float)
        x -= self.center_pixels[0]; y -= self.center_pixels[1]
        angle = 2*np.pi*shift/self.angular_samples
        c, s = np.cos(angle), np.sin(angle)
        u, v = c*x+s*y, -s*x+c*y
        if reflected:
            v = -v
        coords = np.array([v+self.center_pixels[1], u+self.center_pixels[0]])
        return np.stack([map_coordinates(self.image[..., ch], coords, order=1,
                                        mode='grid-constant', prefilter=False)
                         for ch in range(self.channels)], axis=-1)


def evaluate_polar(vertices, target, mask, colors=None, *, color_weight=1.0):
    fail = DomainEvaluation(float('inf'), None, None, MatchMetrics(float('inf'), float('inf'), float('inf')))
    vertices = np.asarray(vertices, float).reshape(-1, 3, 2)
    if not len(vertices) or not np.isfinite(vertices).all() or not np.any(np.asarray(mask) > COVERAGE_EPSILON):
        return fail
    try:
        triangles, _area = centered_triangles(vertices, target.center)
    except ValueError:
        return fail
    use_color = colors is not None and target.has_color and color_weight > 0
    cache = getattr(target, '_polar_fitness_cache', None)
    key = (len(mask), use_color, "opaque-alpha-v1")
    if cache is None:
        cache = {}; target._polar_fitness_cache = cache
    if key not in cache:
        cache[key] = PolarTarget(target, mask, use_color)
    reference = cache[key]
    # Reject invalid out-of-support growth rather than silently dropping it from
    # the image loss. The disk contains the entire original simulation canvas.
    if np.max(np.linalg.norm(triangles-reference.center, axis=-1)) > reference.radius-3/reference.n:
        return fail
    if use_color:
        colors = np.asarray(colors, float)
        if colors.shape != (len(triangles), 3) or not np.isfinite(colors).all():
            return fail
    scaled = (triangles*reference.n+reference.pad)/reference.size
    if use_color:
        density, rgb_density = rasterize_triangles(scaled, reference.size, np.clip(colors, 0, 1))
    else:
        density = rasterize_triangles(scaled, reference.size)
        rgb_density = None
    image = opaque_image(density, rgb_density)
    occupancy = image[..., -1]
    rgb = image[..., :3] if use_color else None
    candidate = reference.sample(sharpen(image))
    losses, shift, reflected, aligned = match_polar(candidate, reference.polar, reference.fft,
                                                    shape_priority=True, color_weight=color_weight)
    total = float(losses[int(reflected), shift])
    posed = reference.posed_image(shift, reflected)
    target_occupancy = posed[..., -1]
    mass = reference.mass
    match = MatchMetrics(float(np.maximum(target_occupancy-occupancy, 0).sum()/mass),
                         float(np.maximum(occupancy-target_occupancy, 0).sum()/mass),
                         float(np.maximum(density-1, 0).sum()/mass))
    shape_energy = max(float(np.square(reference.polar[..., -1]).sum(axis=1).mean()), 1e-12)
    channel_errors = np.square(candidate-aligned).sum(axis=1).mean(axis=0)
    shape_term, color_term = shape_priority_terms(channel_errors, shape_energy, color_weight)
    angle = 2*np.pi*shift/reference.angular_samples
    breakdown = RasterFitnessBreakdown(total, float(shape_term), 0., 0., 0., angle, float(color_term))
    result = DomainEvaluation(total, occupancy, breakdown, match, rgb, target_occupancy,
                              posed[..., :3] if use_color else None)
    result.polar = PolarDiagnostics(reference.polar, candidate, aligned, losses, shift, reflected, reference.radius,
        dict(name='shape-priority-v1', shapeEnergy=shape_energy,
             shapeLoss=float(shape_term), colorContribution=float(color_term),
             colorWeight=float(color_weight), **CONFIG['polarFitness']))
    return result
