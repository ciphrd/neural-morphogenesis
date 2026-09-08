"""Explicit periodic triangle geometry helpers for scene construction/readback."""
import numpy as np

def unwrap_vertices(vertices):
    vertices = np.asarray(vertices,dtype=float).reshape(-1,3,2)
    return vertices[:,0:1]+(vertices-vertices[:,0:1]+.5)%1-.5

def domain_edges(rest):
    rest = np.asarray(rest)
    shape = rest.shape[:-1]
    vertices = unwrap_vertices(rest[...,8:14])
    return (vertices[:,1:]-vertices[:,0:1]).transpose(0,2,1).reshape(*shape,2,2)

def vertices_from_edges(positions, edges):
    positions = np.asarray(positions,dtype=float).reshape(-1,2)
    edges = np.asarray(edges,dtype=float).reshape(-1,2,2)
    a = positions-edges.sum(axis=2)/3
    return (np.stack((a,a+edges[:,:,0],a+edges[:,:,1]),axis=1)%1).astype(np.float32).reshape(-1,6)%1
