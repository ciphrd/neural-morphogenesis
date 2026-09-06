"""Conforming triangle seeds for circular blobs and exact lattice rows.

Both constructions emit twice the material-unit count. Mirrored in rng.ts.
"""
import numpy as np


def triangulate_seed_cells(positions, half_edges):
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
    h = np.asarray(half_edges, dtype=np.float64).reshape(-1, 2, 2)
    if len(h) != len(positions) or np.any(np.linalg.det(h) <= 0):
        raise ValueError('Seed cells must have matching, positively wound domains')
    # Identify lattice corners before conversion to float32. Adjacent cells
    # reuse a single cached coordinate, not separately rounded expressions.
    basis = 2*h[0]
    # Input centers may straddle the torus seam.
    delta = (positions-positions[0]+.5)%1-.5
    sites = np.rint(delta @ np.linalg.inv(basis).T).astype(int)
    if not np.allclose(h,h[0],rtol=1e-6,atol=1e-10):
        raise ValueError('Seed cells must use one lattice basis')
    cache = {}
    def corner(site, sx, sy):
        key = (2*int(site[0])+sx,2*int(site[1])+sy)
        if key not in cache:
            cache[key] = ((positions[0]+basis @ (np.asarray(key)/2))%1).astype(np.float32)%1
        return cache[key]
    triangles = []
    for site in sites:
        a,b,c,d = (corner(site,*sign) for sign in ((-1,-1),(1,-1),(-1,1),(1,1)))
        if np.dot(h[0,:,0],h[0,:,1]) > 0:
            triangles.extend(([a,b,c],[b,d,c]))
        else:
            triangles.extend(([a,b,d],[a,d,c]))
    vertices = np.asarray(triangles,np.float32)
    offsets = (vertices.astype(float)-vertices[:,0:1]+.5)%1-.5
    centers = (vertices[:,0]+offsets.mean(axis=1))%1
    return (centers.astype(np.float32), vertices.reshape(-1,6),
            np.full(2*len(positions), .5, dtype=np.float32))


def triangulate_seed_disk(count, center, spacing, theta):
    """A conforming concentric-ring disk with exactly 2*count triangles.

    Ring populations grow with circumference. The Euler disk identity
    T = 2*interior_vertices + boundary_vertices - 2 fixes the final ring.
    Preserve the legacy packed-cell area, with area-proportional weights.
    A one-cell budget can only represent a quadrilateral (two triangles).
    """
    if count < 1 or int(count) != count or not np.isfinite(spacing) or spacing <= 0:
        raise ValueError('Disk seeds require a positive integer count and spacing')
    total = 2*count
    rings = max(1, int(np.floor(np.sqrt(total/6)+.5)))
    sizes = [int(np.floor(total*k/(rings*rings)+.5)) for k in range(1, rings)]
    sizes.append(total-2*sum(sizes))
    points = [(0., 0.)]
    faces = []
    previous = [0]
    for k, size in enumerate(sizes, 1):
        if count == 1:
            size = 4
        current = []
        for j in range(size):
            angle = theta+2*np.pi*j/size
            current.append(len(points))
            points.append((k/rings*np.cos(angle), k/rings*np.sin(angle)))
        if count == 1:
            faces.extend([(1,2,3), (1,3,4)])
        elif k == 1:
            faces.extend((0,current[j],current[(j+1)%size]) for j in range(size))
        else:
            i = j = 0
            inner = len(previous)
            # Merge angular edge events; integer comparisons make ties portable.
            while i < inner or j < size:
                a, b = previous[i%inner], current[j%size]
                if i < inner and (j == size or (i+1)*size <= (j+1)*inner):
                    faces.append((a,b,previous[(i+1)%inner]))
                    i += 1
                else:
                    faces.append((a,b,current[(j+1)%size]))
                    j += 1
        previous = current
    boundary = len(previous)
    target_area = count*spacing**2*np.sqrt(3)/2
    radius = np.sqrt(target_area/(.5*boundary*np.sin(2*np.pi/boundary)))
    points = (np.asarray(points)*radius+center)%1
    points = points.astype(np.float32)%1
    vertices = points[np.asarray(faces)]
    offsets = (vertices.astype(float)-vertices[:,0:1]+.5)%1-.5
    centers = (vertices[:,0]+offsets.mean(axis=1))%1
    areas = .5*(offsets[:,1,0]*offsets[:,2,1]-offsets[:,1,1]*offsets[:,2,0])
    weights = (count*areas/areas.sum()).astype(np.float32)
    return centers.astype(np.float32), vertices.reshape(-1,6), weights
