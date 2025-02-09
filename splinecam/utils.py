import numpy as np
from scipy.spatial import ConvexHull
import torch

@torch.jit.script
def region_eccentricity_2d(poly: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    dist = torch.pdist(poly[:-1])
    max_len = dist.max()
    min_len = dist.min()
    return max_len / (min_len + eps)

@torch.jit.script
def region_area_2d(poly: torch.Tensor) -> torch.Tensor:
    poly = poly[:-1]
    x, y = poly[:, 0], poly[:, 1]
    S1 = torch.sum(x * torch.roll(y, -1))
    S2 = torch.sum(y * torch.roll(x, -1))
    area = 0.5 * torch.abs(S1 - S2)
    return area

@torch.jit.script
def centrality(polys: torch.Tensor) -> torch.Tensor:
    """
    Calculate mean and standard deviation of distances from vertices to centroid.
    
    Args:
        polys (torch.Tensor): A tensor of shape (n_polys, n_verts, 2) where each poly is a set of vertices.
    
    Returns:
        torch.Tensor: A tensor of shape (2,) containing [mean_dist, std_dist].
    """
    # Flatten all vertices into a single tensor
    verts = torch.cat([poly[:-1] for poly in polys], dim=0)
    centroid = verts.mean(dim=0)
    
    # Compute distances from vertices to centroid
    dist = torch.linalg.norm(verts - centroid, dim=-1)
    
    # Compute mean and std of distances
    mean_dist = dist.mean()
    std_dist = dist.std()
    
    return torch.tensor([mean_dist, std_dist])

@torch.jit.script
def get_region_statistics(polys: torch.Tensor) -> torch.Tensor:
    """
    Calculate various statistics for a list of polygons.
    
    Args:
        polys (torch.Tensor): A tensor of shape (n_polys, n_verts, 2) where each poly is a set of vertices.
    
    Returns:
        torch.Tensor: A tensor of shape (8,) containing the computed statistics.
    """
    # Precompute areas and eccentricities
    areas = torch.stack([region_area_2d(poly) for poly in polys])
    eccs = torch.stack([region_eccentricity_2d(poly) for poly in polys])
    
    # Compute mean and std of areas
    vol_m = areas.mean().item()  # Convert to float
    vol_std = areas.std().item()  # Convert to float
    
    # Compute mean and std of eccentricities
    ecc_m = eccs.mean().item()  # Convert to float
    ecc_std = eccs.std().item()  # Convert to float
    
    # Compute total number of vertices and regions
    nverts = sum((poly.shape[0] - 1) for poly in polys)  # Total vertices
    nregions = len(polys)  # Number of regions
    
    # Compute average number of vertices per region
    avg_verts = nverts / nregions  # Average vertices per region
    
    # Compute centrality metrics
    centr_stats = centrality(polys)  # This returns a tensor [mean_dist, std_dist]
    centr_m = centr_stats[0].item()  # Extract mean distance as float
    centr_std = centr_stats[1].item()  # Extract standard deviation as float
    
    # Combine all statistics into a single tensor
    stats = torch.tensor([
        float(vol_m), float(vol_std),  # Area statistics
        float(nverts), float(nregions),  # Vertex and region counts
        float(ecc_m), float(ecc_std),  # Eccentricity statistics
        float(centr_m), float(centr_std)  # Centrality statistics
    ], dtype=torch.float32)  # Explicitly specify dtype
    
    return stats

@torch.jit.script
def verify_collinear(v_new: torch.Tensor, v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-7) -> bool:
    l1 = torch.linalg.norm(v1 - v2, dim=-1)
    l2 = torch.linalg.norm(v_new - v1, dim=-1)
    l3 = torch.linalg.norm(v_new - v2, dim=-1)
    return torch.allclose(l1, l2 + l3, rtol=0., atol=eps)

@torch.jit.script
def get_Abw(q: torch.Tensor, Wb: torch.Tensor, incoming_Abw: torch.Tensor) -> torch.Tensor:
    activ_Wb = q[..., None] * Wb[None, ...]
    Abw = torch.bmm(activ_Wb[..., :-1], incoming_Abw)
    Abw[..., -1:] = Abw[..., -1:] + activ_Wb[..., -1:]
    return Abw

@torch.jit.script
def get_square_slice_from_one_anchor(
    anchors: torch.Tensor,
    pad_dist: float = 1,
    z1: torch.Tensor = None,
    z2: torch.Tensor = None,
    eps: float = 1e-7
) -> torch.Tensor:
    """
    Create a square slice centered around one anchor point.

    Args:
        anchors (torch.Tensor): A tensor of shape (1, dims).
        pad_dist (float): Padding distance for the slice.
        z1 (torch.Tensor): Precomputed random vector 1.
        z2 (torch.Tensor): Precomputed random vector 2.
        eps (float): Small value to avoid division by zero.

    Returns:
        torch.Tensor: A tensor representing the square slice.
    """
    assert anchors.shape[0] == 1, "Only one anchor point is allowed."
    
    # Use provided random vectors or generate defaults
    if z1 is None:
        z1 = torch.ones_like(anchors[0])  # Default vector (can be replaced with actual random values)
    if z2 is None:
        z2 = torch.zeros_like(anchors[0])  # Default vector (can be replaced with actual random values)

    centroid = anchors[0]
    u1 = z1
    u2 = z2 - (u1 @ z2.T) / (u1 @ u1.T + eps) * u1  # Orthogonalize u2 with respect to u1
    
    dirs = torch.vstack([u1, u2])
    dirs /= torch.linalg.norm(dirs, dim=-1, keepdim=True)  # Normalize directions
    
    domain = torch.vstack([centroid + pad_dist * dirs, centroid - pad_dist * dirs])
    domain_poly = torch.vstack([domain, domain[:1]])  # Close the polygon
    
    return domain_poly

@torch.jit.script
def get_square_slice_from_two_anchors(anchors: torch.Tensor, pad_dist: float = 1, seed: int = -1) -> torch.Tensor:
    if seed != -1:
        torch.manual_seed(seed)
    assert anchors.shape[0] == 2
    centroid = torch.mean(anchors, dim=0)
    u1 = anchors[0] - centroid
    z = torch.randn_like(anchors[0])
    u2 = z - (u1.T @ z) / (u1.T @ u1) * u1
    dirs = torch.vstack([u1, u2])
    dirs /= torch.linalg.norm(dirs, dim=-1, keepdim=True)
    domain = torch.vstack([centroid + pad_dist * dirs, centroid - pad_dist * dirs])
    domain_poly = torch.vstack([domain, domain[:1]])
    return domain_poly

@torch.jit.script
def get_square_slice_from_centroid(anchors: torch.Tensor, pad_dist: float = 1, seed: int = 0, eps: float = 1e-7) -> torch.Tensor:
    assert len(anchors.shape) <= 2 and anchors.shape[0] == 3
    centroid = torch.mean(anchors, dim=0)
    u1 = anchors[0] - centroid
    u2_n = anchors[1] - centroid
    u2 = u2_n - (u1.T @ u2_n) / (u1.T @ u1 + eps) * u1
    dirs = torch.vstack([u1, u2])
    dirs /= torch.linalg.norm(dirs, dim=-1, keepdim=True)
    domain = torch.vstack([centroid + pad_dist * dirs, centroid - pad_dist * dirs])
    domain_poly = torch.vstack([domain, domain[:1]])
    return domain_poly

@torch.jit.script
def get_proj_mat(domain: torch.Tensor) -> torch.Tensor:
    v1 = domain[1] - domain[0]
    v2 = domain[-2] - domain[0]
    v = torch.vstack([v1, v2])
    v /= torch.linalg.norm(v, dim=-1, keepdim=True)
    return torch.hstack([v.T, domain.mean(0, keepdim=True).T])