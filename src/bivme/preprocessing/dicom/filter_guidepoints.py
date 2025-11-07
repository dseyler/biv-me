import os
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger


def fit_plane_to_points(points):
    """
    Fit a plane to a set of 3D points using least squares.
    Returns: (point_on_plane, normal_vector)
    """
    if len(points) < 3:
        raise ValueError("Need at least 3 points to fit a plane")
    
    # Center the points
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    
    # Use SVD to find the plane normal
    U, s, Vt = np.linalg.svd(centered_points)
    # Normal is the last column of V (or last row of Vt)
    normal = Vt[-1, :]
    
    # Normalize the normal vector
    normal = normal / np.linalg.norm(normal)
    
    return centroid, normal


def point_side_of_plane(point, plane_point, plane_normal):
    """
    Determine which side of a plane a point is on.
    Returns: positive value if on one side, negative if on the other.
    """
    vec = point - plane_point
    return np.dot(vec, plane_normal)


def filter_sax_lv_epicardial_guidepoints(output_folder, slice_ids_to_filter=[17, 18], my_logger=None):
    """
    Filter SAX_LV_EPICARDIAL guidepoints for specified slice IDs by removing points
    on the septal side of a plane fitted to first/last RV_septum points from all other SAX slices.
    
    Parameters:
    -----------
    output_folder : str
        Path to folder containing GPFile_*.txt files
    slice_ids_to_filter : list of int
        Slice IDs (frameIDs) to filter (default: [17, 18])
    my_logger : logger
        Logger instance for logging messages
    """
    if my_logger is None:
        my_logger = logger
    
    # Find all GPFiles
    gp_files = sorted(Path(output_folder).glob('GPFile_*.txt'))
    
    if len(gp_files) == 0:
        my_logger.warning(f'No GPFiles found in {output_folder}')
        return
    
    # Process each phase (GPFile) separately
    for gp_file in gp_files:
        my_logger.info(f'Processing {gp_file.name}...')
        
        # Read guidepoint file
        try:
            data = pd.read_csv(gp_file, sep='\t', header=0)
        except Exception as e:
            my_logger.error(f'Error reading {gp_file}: {e}')
            continue
        
        if len(data) == 0:
            my_logger.warning(f'Empty file: {gp_file}')
            continue
        
        # Extract columns
        points = data[['x', 'y', 'z']].values
        contour_types = data['contour type'].values
        frame_ids = data['frameID'].values
        weights = data['weight'].values
        time_frames = data['time frame'].values
        
        # Collect first and last RV_septum points from all SAX slices except filtered ones
        rv_septum_points_for_plane = []
        
        # Get unique SAX slice IDs (excluding filtered ones)
        sax_mask = contour_types == 'SAX_RV_SEPTUM'
        sax_frame_ids = np.unique(frame_ids[sax_mask])
        sax_frame_ids = sax_frame_ids[~np.isin(sax_frame_ids, slice_ids_to_filter)]
        
        for frame_id in sax_frame_ids:
            # Get all RV_septum points for this slice
            mask = (contour_types == 'SAX_RV_SEPTUM') & (frame_ids == frame_id)
            slice_points = points[mask]
            
            if len(slice_points) > 0:
                # Get first and last points (maintaining order as in file)
                first_point = slice_points[0]
                last_point = slice_points[-1]
                rv_septum_points_for_plane.append(first_point)
                rv_septum_points_for_plane.append(last_point)
        
        if len(rv_septum_points_for_plane) < 3:
            my_logger.warning(f'Not enough RV_septum points to fit plane in {gp_file.name}. Skipping filtering.')
            continue
        
        # Fit plane to first/last RV_septum points
        rv_septum_points_for_plane = np.array(rv_septum_points_for_plane)
        try:
            plane_point, plane_normal = fit_plane_to_points(rv_septum_points_for_plane)
        except Exception as e:
            my_logger.error(f'Error fitting plane: {e}')
            continue
        
        # Determine which side is the septal side
        # Get all RV_septum points (from all slices) to determine which side has majority
        all_rv_septum_mask = contour_types == 'SAX_RV_SEPTUM'
        all_rv_septum_points = points[all_rv_septum_mask]
        
        if len(all_rv_septum_points) > 0:
            # Calculate signed distances to plane for all RV_septum points
            signed_distances = np.array([point_side_of_plane(pt, plane_point, plane_normal) 
                                         for pt in all_rv_septum_points])
            
            # The side with the majority of points is the septal side
            # We'll use the sign of the mean to determine the septal side
            mean_dist = np.mean(signed_distances)
            if abs(mean_dist) < 1e-10:  # If mean is essentially zero, use majority count
                positive_count = np.sum(signed_distances > 0)
                negative_count = np.sum(signed_distances < 0)
                septal_side_sign = 1 if positive_count >= negative_count else -1
            else:
                septal_side_sign = np.sign(mean_dist)
        else:
            my_logger.warning(f'No RV_septum points found in {gp_file.name}. Using default side.')
            septal_side_sign = 1  # Default to positive side
        
        # Filter SAX_LV_EPICARDIAL points for slices 17 and 18
        mask_to_keep = np.ones(len(data), dtype=bool)
        
        for slice_id in slice_ids_to_filter:
            # Find SAX_LV_EPICARDIAL points for this slice
            lv_epi_mask = (contour_types == 'SAX_LV_EPICARDIAL') & (frame_ids == slice_id)
            
            if np.any(lv_epi_mask):
                lv_epi_points = points[lv_epi_mask]
                lv_epi_indices = np.where(lv_epi_mask)[0]
                
                # Check which side of plane each point is on
                for i, point in enumerate(lv_epi_points):
                    signed_dist = point_side_of_plane(point, plane_point, plane_normal)
                    
                    # If point is on the septal side (same sign as majority of septum points), remove it
                    if np.sign(signed_dist) == septal_side_sign:
                        mask_to_keep[lv_epi_indices[i]] = False
                
                num_removed = np.sum(~mask_to_keep[lv_epi_indices])
                my_logger.info(f'Removed {num_removed} SAX_LV_EPICARDIAL points from slice {slice_id} in {gp_file.name}')
        
        # Write filtered data back to file
        filtered_data = data[mask_to_keep]
        
        # Write header and data
        with open(gp_file, 'w') as f:
            f.write('x\ty\tz\tcontour type\tframeID\tweight\ttime frame\n')
            for _, row in filtered_data.iterrows():
                f.write('{:.5f}\t{:.5f}\t{:.5f}\t{}\t{}\t{}\t{}\n'.format(
                    row['x'], row['y'], row['z'], row['contour type'], 
                    int(row['frameID']), row['weight'], int(row['time frame'])
                ))
        
        my_logger.success(f'Filtered {gp_file.name}: removed {np.sum(~mask_to_keep)} points total')

