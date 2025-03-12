import cv2
import numba
import numpy as np
from skimage import morphology


def _remove_branching_points(skel: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Remove branching points from a skeletonized image.
    
    Parameters:
    ----------
    
        skel (numpy.ndarray): Binary skeletonized image where the skeleton is represented by 1s.
        kernel_size (int, optional): Size of the kernel used to count neighbours. Default is 3.
    
    Returns:
    -------
        numpy.ndarray: Skeletonized image with branching points removed.
    """
    
    # Define kernel
    kernel = np.ones((kernel_size, kernel_size))
    
    # Ensure binary image (0s and 1s only) as float
    skel = np.clip(skel, 0, 1, dtype=np.float32)
    
    # Count neighbours for each pixel
    im_filt = cv2.filter2D(skel, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    im_filt = im_filt * skel # Remove non-skeleton pixels
    
    # Identify branching points (3+ neighbours)
    branch_points = im_filt > 3
    
    # Remove branching points
    skel_nbp = skel * ~branch_points
    
    return skel_nbp

def _remove_small_branches(skel: np.ndarray, min_length: int = 10) -> np.ndarray:
    """
    Remove small branches from skeleton (sparse tensor).
    
    Parameters:
    ----------
    
        skel (np.ndarray): Labeled skeleton tensor without branching points of shape (H,W).
        min_length (int, optional): Minimum length of branch to keep. Default is 10.
    
    Returns:
    -------
        torch.Tensor: Skeleton tensor with small branches removed.
    """
    # Flatten to compute counts of each label
    flattened = skel.flatten()
    counts = np.bincount(flattened)

    # Ignore background label (often 0)
    counts[0] = 0

    # Boolean mask for labels to keep
    keep_mask = (counts >= min_length)

    # Create a mapping array that re-labels only the valid labels
    mapping = np.zeros_like(counts)
    labels_to_keep = np.nonzero(keep_mask)[0]
    mapping[labels_to_keep] = np.arange(1, labels_to_keep.size + 1)

    # Remap the original skeleton in one pass
    skel_nsb = mapping[skel]

    return skel_nsb


def _reorder_coords(skel: np.ndarray, origin: tuple) -> list[np.ndarray]:
    """Reorder coordinates of skeleton w.r.t. origin.
    
    Args:
        skel: Skeleton tensor of shape (H,W,C)
        origin: Origin coordinates (x,y)
    Returns:
        list[np.ndarray]: List of ordered paths.
    """
    
    def _reorder_all_in_one_pass(
        skel: np.ndarray, 
        start_coords: np.ndarray
        ) -> list[np.ndarray]:
        """
        Wraps everything up:
        1) Extract coords from a single channel skeleton containing multiple disjoint linear paths.
        2) Build a global adjacency matrix for all coords.
        3) Find connected components, reorder each from start → end, return them all.
        """
        
        def _dfs_order_path(coords_2d, adjacency_2d, start_idx):
            """
            Perform a simple DFS from start_idx to produce a linear ordering
            of coords_2d for a strictly linear path.
            """
            visited_local = np.zeros(coords_2d.shape[0], dtype=bool)
            order = []
            stack = [start_idx]
            while stack:
                current = stack.pop()
                if visited_local[current]:
                    continue
                visited_local[current] = True
                order.append(current)
                # push neighbors
                nbrs = adjacency_2d[current].nonzero()[0]
                for nbr in nbrs:
                    if not visited_local[nbr]:
                        stack.append(nbr)

            # Convert indices back to coordinates
            return coords_2d[order]
        
        def _reorder_multiple_paths(
            coords: np.ndarray, 
            adjacency: np.ndarray, 
            start_indices: list[int]
        ) -> list[np.ndarray]:
            """
            Reorder multiple 8-connected paths by only looping over the known start indices,
            rather than every pixel in `coords`. Each path is discovered by DFS from its start.
            
            Args:
                coords (np.ndarray): (N, 2) array of all 'on' pixels.
                adjacency (np.ndarray): (N, N) boolean adjacency for 8-connected neighbors.
                start_indices (list[int]): Indices in `coords` for each path's known start pixel.
            
            Returns:
                list[np.ndarray]: A list of reordered path coordinates, one per start index.
            """
            visited = np.zeros(coords.shape[0], dtype=bool)
            paths = []

            for s in start_indices:
                if visited[s]:
                    # Already discovered in a previous DFS
                    continue
                # DFS to find all pixels in this connected component
                stack = [s]
                component_indices = []
                
                while stack:
                    current = stack.pop()
                    if visited[current]:
                        continue
                    visited[current] = True
                    component_indices.append(current)
                    # Traverse neighbors
                    neighbors = adjacency[current].nonzero()[0]
                    for nbr in neighbors:
                        if not visited[nbr]:
                            stack.append(nbr)
                
                # Now `component_indices` holds all pixels in this path's connected component
                if len(component_indices) == 0:
                    # No path discovered from this start
                    continue

                component_coords = coords[component_indices]

                # For a strictly linear path, find endpoints or just reorder from the known start.
                # Build adjacency for the component alone:
                comp_row_diff = np.abs(component_coords[:, 0:1] - component_coords[:, 0])
                comp_col_diff = np.abs(component_coords[:, 1:2] - component_coords[:, 1])
                comp_adj = (comp_row_diff <= 1) & (comp_col_diff <= 1) & (
                    ~((comp_row_diff == 0) & (comp_col_diff == 0))
                )

                # In a strictly linear path, exactly two endpoints will have 1 neighbor
                neighbor_counts = np.sum(comp_adj, axis=1)
                endpoints = np.where(neighbor_counts == 1)[0]
                if len(endpoints) == 2:
                    local_start_idx = endpoints[0]
                else:
                    # fallback if no distinct endpoints
                    local_start_idx = 0
                
                path_sorted = _dfs_order_path(component_coords, comp_adj, local_start_idx)
                paths.append(path_sorted)

            return paths
        
        @numba.njit()
        def _build_adjacency(coords):
            N = coords.shape[0]
            adjacency = np.zeros((N, N), dtype=np.bool_)
            for i in numba.prange(N):
                for j in range(N):
                    if i == j:
                        continue
                    # Check if coords are within 1 row and 1 col
                    if (abs(coords[i,0] - coords[j,0]) < 2 and 
                        abs(coords[i,1] - coords[j,1]) < 2):
                        adjacency[i, j] = True
            return adjacency
        
        # 1) Extract all 'on' pixel coords
        coords = np.argwhere(skel > 0)  # shape (N, 2)
        start_idx = [np.where((coords == start).all(axis=1))[0][0] for start in start_coords]
        
        # 2) Build NxN adjacency with broadcasting
        adjacency = _build_adjacency(coords)
        
        # 3) Reorder each connected component
        paths = _reorder_multiple_paths(coords, adjacency, start_idx)
        
        return paths
    
    def _get_endpoints(skel: np.ndarray) -> np.ndarray:
        """Get endpoint coordinates of skeleton.
        
        Args:
            skel: Skeleton of shape (H,W)
        Returns:
            np.ndarray: Endpoint coordinates of shape (N,3)
        """
        
        # Define convolution kernel
        kernel = np.ones((3,3), dtype=np.uint8)
        kernel[1,1,...] = 10
        
        # Convolve skeleton with kernel
        _skel = skel.astype(bool).astype(np.float32) # Turn all nonzero values of skel to 1
        skel_conv = cv2.filter2D(_skel, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        
        # Keep only endpoints
        endpoints = (skel_conv == np.uint8(11))
        endpoints = skel * endpoints # Keep the values of skel where endpoints is True
        
        # Get coordinates indexes (row, col)
        coords = np.nonzero(endpoints)
        
        # Add pixel value as 3rd coordinate
        coords = np.stack(coords, axis=1)
        coords = coords.reshape(-1,2)
        value = endpoints[coords[:,0], coords[:,1]]
        coords = np.concatenate([coords, value[:,np.newaxis]], axis=1)
            
        return coords

     # Get endpoints
    endpoints = _get_endpoints(skel) # [[x_i,y_i,c_i],...]
    
    # Transform origin to match endpoints shape
    origin = np.array(origin).reshape(1,2)
    origin = np.repeat(origin, endpoints.shape[0], axis=0)
    
    # Sort endpoints by distance to origin
    # print(np.linalg.norm(endpoints[:,0:2] - origin, axis=1))
    distances = np.linalg.norm(endpoints[:,0:2] - origin, axis=1)
    # distances = np.sum((endpoints[:,0:1] - origin).astype(np.float32)**2, axis=1)
    idx = np.argsort(distances)
    endpoints = endpoints[idx]
    
    # Re-sort by channel without modifying the order of the other columns
    # endpoints = endpoints[endpoints[:,2].argsort(stable=True)]]
    endpoints = endpoints[np.argsort(endpoints[:,2], kind='stable')] # incorrect
    # endpoints = endpoints[np.argsort(endpoints[:,2])] # incorrect
    start = endpoints[::2, :2]
    
    # Find paths
    paths = _reorder_all_in_one_pass(skel, start)
    
    return paths


    
def generate_vessel_skeleton(vessels, od_mask, od_centre, min_length=10) -> np.ndarray:
    
    # Remove optic disc
    vessels[od_mask > 0] = 0
    
    # Close vessels
    filt = morphology.disk(3)
    v_small = cv2.morphologyEx(vessels.astype(np.uint8), cv2.MORPH_CLOSE, filt)
    
    # Skeletonize using OpenCV
    v_skel_all = morphology.skeletonize(v_small)
    
    # Remove branching points
    v_skel = _remove_branching_points(v_skel_all)
    v_skel_labels = morphology.label(v_skel)
    
    # Remove small branches (less than 10 pixels)
    v_skel = _remove_small_branches(v_skel_labels, min_length)
    
    # Reorder coordinates w.r.t. OD center
    coords_sorted = _reorder_coords(v_skel, od_centre)
    
    return coords_sorted