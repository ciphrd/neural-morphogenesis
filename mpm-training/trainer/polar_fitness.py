"""IsoNCA image loss: sharpen, polar sample, minimize squared error over O(2).

Reference: google-research/self-organising-systems/isotropic_nca/
blogpost_isonca_single_seed_pytorch.ipynb (InvariantLoss and sharpen_filter).
We use an exactly periodic angular lattice. Radius covers the entire Cartesian
canvas, not just its inscribed disk, so corner details cannot disappear. There
is no radial Jacobian weighting: like the reference, sum angles, mean radii and
channels. The fixed sampling grid depends only on the target and resolution.
"""
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

    @property
    def angle(self):
        return 2*np.pi*self.shift/self.target.shape[1]


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


def match_polar(candidate, target, target_fft=None):
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
    losses = np.stack([(norms[:, None, :]-2*c).mean(axis=(0, 2)) for c in correlations])
    losses = np.maximum(losses, 0.)  # remove cancellation roundoff near exact matches
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
        image = mask[..., None] if rgb is None else np.concatenate([rgb, mask[..., None]], axis=-1)
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


def evaluate_polar(vertices, target, mask, colors=None):
    fail = DomainEvaluation(float('inf'), None, None, MatchMetrics(float('inf'), float('inf'), float('inf')))
    vertices = np.asarray(vertices, float).reshape(-1, 3, 2)
    if not len(vertices) or not np.isfinite(vertices).all() or not mask.any():
        return fail
    try:
        triangles, _area = centered_triangles(vertices, target.center)
    except ValueError:
        return fail
    use_color = colors is not None and target.has_color
    cache = getattr(target, '_polar_fitness_cache', None)
    key = (len(mask), use_color)
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
        rgb = rgb_density/np.maximum(density[..., None], 1.)
    else:
        density = rasterize_triangles(scaled, reference.size)
        rgb = None
    occupancy = np.clip(density, 0, 1)
    image = occupancy[..., None] if rgb is None else np.concatenate([rgb, occupancy[..., None]], axis=-1)
    candidate = reference.sample(sharpen(image))
    losses, shift, reflected, aligned = match_polar(candidate, reference.polar, reference.fft)
    total = float(losses[int(reflected), shift])
    posed = reference.posed_image(shift, reflected)
    target_occupancy = posed[..., -1]
    mass = float(mask.sum())
    match = MatchMetrics(float(np.maximum(target_occupancy-occupancy, 0).sum()/mass),
                         float(np.maximum(occupancy-target_occupancy, 0).sum()/mass),
                         float(np.maximum(density-1, 0).sum()/mass))
    terms = np.square(candidate-aligned).sum(axis=1).mean(axis=0)/reference.channels
    angle = 2*np.pi*shift/reference.angular_samples
    breakdown = RasterFitnessBreakdown(total, float(terms[-1]), 0., 0., 0., angle, float(terms[:-1].sum()))
    result = DomainEvaluation(total, occupancy, breakdown, match, rgb, target_occupancy,
                              posed[..., :3] if use_color else None)
    result.polar = PolarDiagnostics(reference.polar, candidate, aligned, losses, shift, reflected, reference.radius)
    return result
