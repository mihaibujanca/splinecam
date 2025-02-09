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
def get_region_means(regions: torch.Tensor, dims: int, dtype: torch.dtype = torch.float64, device: str = 'cuda') -> torch.Tensor:
    """
    Compute the means of each region.

    Args:
        regions (torch.Tensor): A tensor of shape (n_regions, n_verts, dims).
        dims (int): The dimensionality of the points.
        dtype (torch.dtype): Desired data type for the output tensor.
        device (str): Device on which the tensor resides.

    Returns:
        torch.Tensor: A tensor of shape (n_regions, dims) containing the means.
    """
    n_regions = regions.shape[0]
    means = torch.zeros(n_regions, dims, dtype=dtype, device=device)

    for i in range(n_regions):
        means[i] = regions[i].mean(dim=0)

    return means

@torch.jit.script
def regions_list2vec(regions: torch.Tensor, repeat_first: bool = True) -> tuple:
    """
    Convert a list of cycles into a single vector and indices.

    Args:
        regions (torch.Tensor): A tensor of shape (n_regions, n_verts, dims).
        repeat_first (bool): Whether to repeat the first vertex of each region.

    Returns:
        tuple: A tuple containing:
            - out_cycles (torch.Tensor): A tensor of shape (total_verts, dims).
            - cyc_idx (torch.Tensor): A tensor of shape (total_verts,) with cycle indices.
            - ends (torch.Tensor): A tensor of shape (n_regions,) with end indices.
    """
    if repeat_first:
        regions = torch.cat([regions, regions[:, :1]], dim=1)

    out_cycles = torch.vstack([region for region in regions])
    cyc_idx = torch.zeros(out_cycles.shape[0], dtype=torch.int64)
    ends = torch.zeros(len(regions), dtype=torch.int64)

    start = 0
    for i in range(len(regions)):
        n = regions[i].shape[0]
        cyc_idx[start:start + n] = i
        start += n
        ends[i] = start

    return out_cycles, cyc_idx, ends
@torch.jit.script
def split_domain_by_edge(domain: torch.Tensor) -> torch.Tensor:
    """
    Split a domain into edge-centroid polygons.

    Args:
        domain (torch.Tensor): A tensor of shape (n_verts, dims).

    Returns:
        torch.Tensor: A tensor of shape (n_edges, 4, dims) representing the split polygons.
    """
    centroid = domain[:-1].mean(dim=0)
    out_domain = []

    for i in range(len(domain) - 1):
        each1, each2 = domain[i], domain[i + 1]
        poly = torch.stack([each1, each2, centroid, each1])
        out_domain.append(poly)

    return torch.stack(out_domain)

@torch.no_grad()
@torch.jit.script
def get_nneigh_points(data1: torch.Tensor, data2: torch.Tensor) -> torch.Tensor:
    """
    Find nearest neighbors between two datasets.

    Args:
        data1 (torch.Tensor): Tensor of shape (n1, dims).
        data2 (torch.Tensor): Tensor of shape (n2, dims).

    Returns:
        torch.Tensor: A tensor of shape (3, dims) containing the nearest points.
    """
    dist = torch.cdist(data1, data2)
    idx1 = torch.argsort(dist.min(dim=1).values)
    idx2 = torch.argsort(dist[idx1[0]])

    points = torch.vstack([data1[idx1[0]], data2[idx2[:2]]])
    return points

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