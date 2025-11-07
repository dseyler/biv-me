"""
Temporal smoothing of fitted biventricular models across cardiac phases.

This module provides functions to apply Fourier smoothing to control mesh
coordinates across time to reduce jerkiness in fitted models. Smoothing
strength varies per control point based on distance to guidepoints.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import fft
from scipy.spatial import cKDTree
from loguru import logger
from copy import deepcopy

from bivme.fitting.BiventricularModel import BiventricularModel
from bivme.fitting.GPDataSet import GPDataSet
from bivme.fitting.surface_enum import Surface, ControlMesh
from bivme.meshing.mesh_io import write_vtk_surface, export_to_obj
from bivme import MODEL_RESOURCE_DIR


def smooth_control_meshes_temporal(
    output_folder: Path,
    case: str,
    gp_suffix: str,
    gp_folder,
    filename_info,
    si_suffix: str,
    config: dict,
    output_format: str,
    my_logger: logger = logger,
    vtk_output_subdir: str = None,
) -> None:
    """
    Apply temporal smoothing to control mesh coordinates across cardiac phases.
    
    This function reads fitted model files, computes distances from control points
    to guidepoints, applies variable Fourier smoothing based on these distances,
    and writes smoothed models. Original unsmoothed models are preserved.
    
    Parameters
    ----------
    output_folder : Path
        Directory containing the fitted model files
    case : str
        Case name/identifier
    gp_suffix : str
        Suffix used in GPFile names
    gp_folder : Path
        Directory containing guidepoint files
    filename_info : Path
        Path to SliceInfoFile
    si_suffix : str
        Suffix for SliceInfoFile
    config : dict
        Configuration dictionary containing temporal_smoothing parameters
    output_format : str
        Output format (".vtk", ".obj", or "none")
    my_logger : logger
        Logger instance for logging messages
    vtk_output_subdir : str, optional
        Subdirectory name for VTK output (e.g., "vtk_PA" for refitted models).
        If None, uses default directory structure based on output_format.
    """
    smoothing_config = config.get("temporal_smoothing", {})
    use_distance_based = smoothing_config.get("use_distance_based_smoothing", True)
    min_cutoff = smoothing_config.get("min_cutoff_frequency", 0.3)
    max_cutoff = smoothing_config.get("max_cutoff_frequency", 0.8)
    char_length = smoothing_config.get("characteristic_length", 5.0)
    base_cutoff = smoothing_config.get("cutoff_frequency", 0.8)
    
    # Ensure Path objects
    output_folder = Path(output_folder)
    gp_folder = Path(gp_folder)
    filename_info = Path(filename_info)
    
    # Find all model files
    model_files = sorted(
        output_folder.glob(f"{case}{gp_suffix}_model_frame_*.txt")
    )
    
    if len(model_files) == 0:
        my_logger.warning(f"No model files found for temporal smoothing in {output_folder}")
        return
    
    # Extract frame numbers and sort
    frame_numbers = []
    for f in model_files:
        frame_match = f.stem.split("_frame_")[-1]
        frame_numbers.append(int(frame_match))
    
    # Sort by frame number
    sorted_indices = np.argsort(frame_numbers)
    model_files = [model_files[i] for i in sorted_indices]
    frame_numbers = [frame_numbers[i] for i in sorted_indices]
    
    my_logger.info(f"Reading {len(model_files)} model files for temporal smoothing...")
    
    # Read all control meshes
    control_meshes = []
    num_control_points = None
    
    for model_file in model_files:
        try:
            df = pd.read_csv(model_file, sep=",")
            # Extract x, y, z coordinates
            control_mesh = np.column_stack([df["x"].values, df["y"].values, df["z"].values])
            if num_control_points is None:
                num_control_points = len(control_mesh)
            elif len(control_mesh) != num_control_points:
                my_logger.warning(
                    f"Inconsistent number of control points in {model_file}: "
                    f"expected {num_control_points}, got {len(control_mesh)}"
                )
                continue
            control_meshes.append(control_mesh)
        except Exception as e:
            my_logger.error(f"Error reading {model_file}: {e}")
            continue
    
    if len(control_meshes) == 0:
        my_logger.error("No valid control meshes found for smoothing")
        return
    
    # Convert to array: (num_phases, 388, 3)
    control_meshes = np.array(control_meshes)
    num_phases = len(control_meshes)
    
    my_logger.info(f"Applying Fourier smoothing to {num_phases} phases with {num_control_points} control points...")
    
    # Compute distances from control points to guidepoints if distance-based smoothing is enabled
    cutoff_frequencies = None
    if use_distance_based:
        my_logger.info("Computing distances from control points to guidepoints...")
        avg_distances = compute_control_point_distances(
            output_folder, case, gp_suffix, frame_numbers, gp_folder, 
            filename_info, si_suffix, config, my_logger
        )
        
        if avg_distances is not None:
            # Map distances to cutoff frequencies using sigmoid transition
            cutoff_frequencies = distance_to_cutoff(
                avg_distances, min_cutoff, max_cutoff, char_length, my_logger
            )
            my_logger.info(
                f"Distance-based smoothing: cutoff frequencies range from "
                f"{cutoff_frequencies.min():.3f} to {cutoff_frequencies.max():.3f}"
            )
        else:
            my_logger.warning("Could not compute distances, using uniform smoothing")
            use_distance_based = False
    
    # Apply smoothing
    if use_distance_based and cutoff_frequencies is not None:
        smoothed = apply_variable_fourier_smoothing(
            control_meshes, cutoff_frequencies, my_logger
        )
    else:
        # Use uniform cutoff frequency
        smoothed = apply_fourier_smoothing(
            control_meshes, base_cutoff, my_logger
        )
    
    # Write smoothed models
    my_logger.info("Writing smoothed model files...")
    for i, frame_num in enumerate(frame_numbers):
        smoothed_file = output_folder / f"{case}{gp_suffix}_model_frame_{frame_num:03}_smoothed.txt"
        write_model_file(smoothed_file, smoothed[i], frame_num)
    
    # Generate VTK files for smoothed models if output format is not "none"
    if output_format != "none":
        my_logger.info("Generating VTK files for smoothed models...")
        
        # Load base model for mesh structure
        base_model = BiventricularModel(MODEL_RESOURCE_DIR, case)
        
        # Determine VTK output directory and suffix
        # If vtk_output_subdir is specified, use parent directory + subdir (e.g., for refitted models)
        # and don't add suffix to filenames (overwrite existing files)
        if vtk_output_subdir:
            vtk_output_folder = output_folder.parent / vtk_output_subdir
            vtk_suffix = ""  # Overwrite existing files, no suffix
        else:
            vtk_output_folder = output_folder
            vtk_suffix = "_smoothed"  # Add suffix for original models
        
        # Check if merged mesh should be generated
        # Only for smoothed refitted models (vtk_output_subdir == "vtk_PA")
        # and only if merged_mesh = true and all three surfaces are in output_meshes
        generate_merged = False
        if vtk_output_subdir == "vtk_PA":
            required_surfaces = {"LV_ENDOCARDIAL", "RV_ENDOCARDIAL", "EPICARDIAL"}
            output_meshes_set = set(config["output_fitting"]["output_meshes"])
            if (config["output_fitting"].get("merged_mesh", False) and 
                required_surfaces.issubset(output_meshes_set)):
                generate_merged = True
        
        # Generate VTK for smoothed models
        generate_vtk_for_smoothed_models(
            vtk_output_folder, case, gp_suffix, frame_numbers, smoothed,
            base_model, config, output_format, vtk_suffix, my_logger,
            generate_merged_mesh=generate_merged
        )
    
    my_logger.success(
        f"Temporal smoothing completed: {len(frame_numbers)} phases smoothed"
    )


def compute_control_point_distances(
    output_folder,
    case: str,
    gp_suffix: str,
    frame_numbers: list,
    gp_folder,
    filename_info,
    si_suffix: str,
    config: dict,
    my_logger: logger,
) -> np.ndarray:
    """
    Compute average distance from each control point to nearest guidepoint across all phases.
    
    For each frame, loads the fitted model and guidepoints, computes distances from
    control points (via subdivision surface) to guidepoints, and averages across frames.
    
    Parameters
    ----------
    output_folder : Path
        Directory containing fitted model files
    case : str
        Case name
    gp_suffix : str
        GP suffix
    frame_numbers : list
        List of frame numbers
    gp_folder : Path
        Directory containing guidepoint files
    filename_info : Path
        Path to SliceInfoFile
    si_suffix : str
        Suffix for SliceInfoFile
    config : dict
        Configuration dictionary
    my_logger : logger
        Logger instance
        
    Returns
    -------
    np.ndarray
        Array of shape (num_control_points,) with average distances, or None if computation fails
    """
    # Ensure Path objects
    output_folder = Path(output_folder)
    gp_folder = Path(gp_folder)
    filename_info = Path(filename_info)
    
    num_control_points = 388  # BiventricularModel.NUM_NODES
    distances_per_frame = []
    
    for frame_num in frame_numbers:
        # Load fitted model
        model_file = output_folder / f"{case}{gp_suffix}_model_frame_{frame_num:03}.txt"
        if not model_file.exists():
            my_logger.warning(f"Model file not found: {model_file}, skipping frame {frame_num}")
            continue
        
        try:
            # Read control mesh
            df = pd.read_csv(model_file, sep=",")
            control_mesh = np.column_stack([df["x"].values, df["y"].values, df["z"].values])
            
            # Create model instance with fitted control mesh
            model = BiventricularModel(MODEL_RESOURCE_DIR, case)
            model.update_control_mesh(control_mesh)
            
            # Get surface points (subdivided from control mesh)
            surface_points = model.et_pos  # Shape: (5810, 3)
            
            # Load guidepoints for this frame
            gp_file = gp_folder / f"GPFile_{gp_suffix}{frame_num:03}.txt"
            if not gp_file.exists():
                my_logger.warning(f"Guidepoint file not found: {gp_file}, skipping frame {frame_num}")
                continue
            
            gp_dataset = GPDataSet(
                str(gp_file),
                str(filename_info),
                case,
                sampling=config.get("gp_processing", {}).get("sampling", 1),
                time_frame_number=frame_num,
            )
            
            if not gp_dataset.success:
                my_logger.warning(f"Failed to load GPDataSet for frame {frame_num}")
                continue
            
            guidepoints = gp_dataset.points_coordinates  # Shape: (N, 3)
            
            if len(guidepoints) == 0:
                my_logger.warning(f"No guidepoints found for frame {frame_num}")
                continue
            
            # Build KD-tree for guidepoints
            guidepoint_tree = cKDTree(guidepoints)
            
            # For each control point, find distance to nearest guidepoint
            # Since control mesh is fitted to guidepoints, we can compute distance directly
            # from control point coordinates to guidepoints
            control_distances = np.zeros(num_control_points)
            
            for ctrl_idx in range(num_control_points):
                # Get control point position
                ctrl_point = control_mesh[ctrl_idx, :].reshape(1, -1)
                
                # Find distance to nearest guidepoint
                distances, _ = guidepoint_tree.query(ctrl_point, k=1)
                control_distances[ctrl_idx] = distances[0]
            
            distances_per_frame.append(control_distances)
            
        except Exception as e:
            my_logger.error(f"Error computing distances for frame {frame_num}: {e}")
            continue
    
    if len(distances_per_frame) == 0:
        my_logger.error("Could not compute distances for any frame")
        return None
    
    # Average distances across frames
    avg_distances = np.mean(distances_per_frame, axis=0)
    
    my_logger.info(
        f"Computed average distances: min={avg_distances.min():.2f}mm, "
        f"max={avg_distances.max():.2f}mm, mean={avg_distances.mean():.2f}mm"
    )
    
    return avg_distances


def distance_to_cutoff(
    distances: np.ndarray,
    min_cutoff: float,
    max_cutoff: float,
    char_length: float,
    my_logger: logger,
) -> np.ndarray:
    """
    Map distances to cutoff frequencies using sigmoid transition.
    
    Close points (distance=0) get higher cutoff (less smoothing).
    Far points (distance→∞) get lower cutoff (more smoothing).
    
    Uses sigmoid function: cutoff(d) = max_cutoff - (max_cutoff - min_cutoff) * sigmoid(d / L)
    where sigmoid(x) = 1 / (1 + exp(-x))
    
    This ensures:
    - Close points (d≈0): cutoff ≈ max_cutoff (less smoothing, higher frequency)
    - Far points (d→∞): cutoff → min_cutoff (more smoothing, lower frequency)
    
    Parameters
    ----------
    distances : np.ndarray
        Array of distances (num_control_points,)
    min_cutoff : float
        Minimum cutoff frequency (for far points, more smoothing)
    max_cutoff : float
        Maximum cutoff frequency (for close points, less smoothing)
    char_length : float
        Characteristic length (in pixels) for transition
    my_logger : logger
        Logger instance
        
    Returns
    -------
    np.ndarray
        Array of cutoff frequencies (num_control_points,)
    """
    # Normalize distances by characteristic length
    normalized_dist = distances / char_length
    
    # Apply sigmoid function: sigmoid(x) = 1 / (1 + exp(-x))
    # For x=0: sigmoid(0) ≈ 0.5
    # For x→∞: sigmoid(∞) = 1
    # For x→-∞: sigmoid(-∞) = 0
    
    # We want: cutoff(0) = max_cutoff (close points, less smoothing)
    #          cutoff(∞) = min_cutoff (far points, more smoothing)
    # So: cutoff(d) = max_cutoff - (max_cutoff - min_cutoff) * sigmoid(d/L)
    sigmoid = 1.0 / (1.0 + np.exp(-normalized_dist))
    cutoff_freqs = max_cutoff - (max_cutoff - min_cutoff) * sigmoid
    
    return cutoff_freqs


def apply_variable_fourier_smoothing(
    control_meshes: np.ndarray,
    cutoff_frequencies: np.ndarray,
    my_logger: logger,
) -> np.ndarray:
    """
    Apply Fourier smoothing with variable cutoff frequency per control point.
    
    Parameters
    ----------
    control_meshes : np.ndarray
        Array of shape (num_phases, num_control_points, 3)
    cutoff_frequencies : np.ndarray
        Array of shape (num_control_points,) with cutoff frequency for each point
    my_logger : logger
        Logger instance
        
    Returns
    -------
    np.ndarray
        Smoothed control meshes with same shape as input
    """
    num_phases, num_points, num_coords = control_meshes.shape
    smoothed = np.zeros_like(control_meshes)
    
    for point_idx in range(num_points):
        cutoff_freq = cutoff_frequencies[point_idx]
        num_freqs = max(1, int(num_phases * cutoff_freq))
        
        for coord_idx in range(num_coords):
            # Extract temporal sequence for this point and coordinate
            temporal_seq = control_meshes[:, point_idx, coord_idx]
            
            # Apply FFT
            fft_result = fft.fft(temporal_seq)
            
            # Apply low-pass filter: keep only first num_freqs frequencies
            filtered_fft = np.zeros_like(fft_result)
            filtered_fft[:num_freqs] = fft_result[:num_freqs]
            if num_phases > 2 * num_freqs:
                # Keep symmetric frequencies for real signal
                filtered_fft[-num_freqs+1:] = fft_result[-num_freqs+1:]
            
            # Inverse FFT
            smoothed_seq = np.real(fft.ifft(filtered_fft))
            smoothed[:, point_idx, coord_idx] = smoothed_seq
    
    return smoothed


def apply_fourier_smoothing(
    control_meshes: np.ndarray, cutoff_freq: float, my_logger: logger
) -> np.ndarray:
    """
    Apply uniform Fourier smoothing to control mesh coordinates.
    
    Parameters
    ----------
    control_meshes : np.ndarray
        Array of shape (num_phases, num_control_points, 3)
    cutoff_freq : float
        Fraction of Nyquist frequency to keep (0.0 to 1.0)
    my_logger : logger
        Logger instance
        
    Returns
    -------
    np.ndarray
        Smoothed control meshes with same shape as input
    """
    num_phases, num_points, num_coords = control_meshes.shape
    smoothed = np.zeros_like(control_meshes)
    
    # Calculate number of frequencies to keep
    num_freqs = max(1, int(num_phases * cutoff_freq))
    
    for point_idx in range(num_points):
        for coord_idx in range(num_coords):
            # Extract temporal sequence for this point and coordinate
            temporal_seq = control_meshes[:, point_idx, coord_idx]
            
            # Apply FFT
            fft_result = fft.fft(temporal_seq)
            
            # Apply low-pass filter: keep only first num_freqs frequencies
            # (plus symmetric frequencies for real signal)
            filtered_fft = np.zeros_like(fft_result)
            filtered_fft[:num_freqs] = fft_result[:num_freqs]
            if num_phases > 2 * num_freqs:
                # Keep symmetric frequencies for real signal
                filtered_fft[-num_freqs+1:] = fft_result[-num_freqs+1:]
            
            # Inverse FFT
            smoothed_seq = np.real(fft.ifft(filtered_fft))
            smoothed[:, point_idx, coord_idx] = smoothed_seq
    
    return smoothed


def write_model_file(filepath: Path, control_mesh: np.ndarray, frame_num: int) -> None:
    """
    Write control mesh to model file in the same format as original.
    
    Parameters
    ----------
    filepath : Path
        Path to output file
    control_mesh : np.ndarray
        Control mesh array of shape (num_control_points, 3)
    frame_num : int
        Frame number
    """
    model_data = {
        "x": control_mesh[:, 0],
        "y": control_mesh[:, 1],
        "z": control_mesh[:, 2],
        "Frame": [frame_num] * len(control_mesh),
    }
    model_data_frame = pd.DataFrame(data=model_data)
    with open(filepath, "w") as file:
        file.write(
            model_data_frame.to_csv(
                header=True, index=False, sep=",", lineterminator="\n"
            )
        )


def generate_vtk_for_smoothed_models(
    output_folder: Path,
    case: str,
    gp_suffix: str,
    frame_numbers: list,
    smoothed_meshes: np.ndarray,
    base_model: BiventricularModel,
    config: dict,
    output_format: str,
    suffix: str,
    my_logger: logger,
    generate_merged_mesh: bool = False,
) -> None:
    """
    Generate VTK files for smoothed models using the same structure as original.
    
    Parameters
    ----------
    output_folder : Path
        Output directory
    case : str
        Case name
    gp_suffix : str
        GP suffix
    frame_numbers : list
        List of frame numbers
    smoothed_meshes : np.ndarray
        Array of smoothed control meshes (num_phases, 388, 3)
    base_model : BiventricularModel
        Base model instance for mesh structure
    config : dict
        Configuration dictionary
    output_format : str
        Output format (".vtk" or ".obj")
    suffix : str
        Suffix to add to output directory (e.g., "_smoothed")
    my_logger : logger
        Logger instance
    generate_merged_mesh : bool
        Whether to generate merged mesh (LV_ENDOCARDIAL + RV_ENDOCARDIAL + EPICARDIAL)
        excluding valve surfaces. Only applies to smoothed refitted models.
    """
    # Create output directory
    # If suffix is empty, output_folder is already the target directory (e.g., vtk_PA)
    # Otherwise, create subdirectory with suffix (e.g., vtk_smoothed)
    if output_format == ".vtk":
        if suffix:
            vtk_dir = output_folder / f"vtk{gp_suffix}{suffix}"
        else:
            vtk_dir = output_folder  # Use output_folder directly (already vtk_PA)
        vtk_dir.mkdir(exist_ok=True)
    elif output_format == ".obj":
        if suffix:
            obj_dir = output_folder / f"obj{gp_suffix}{suffix}"
        else:
            obj_dir = output_folder  # Use output_folder directly
        obj_dir.mkdir(exist_ok=True)
    else:
        return
    
    # Build meshes dictionary (same as in perform_fitting)
    meshes = {}
    for surface in Surface:
        mesh_data = {}
        if surface.name in config["output_fitting"]["output_meshes"]:
            mesh_data[surface.name] = surface.value
            if surface.name == "LV_ENDOCARDIAL" and config["output_fitting"]["closed_mesh"] == True:
                mesh_data["MITRAL_VALVE"] = Surface.MITRAL_VALVE.value
                mesh_data["AORTA_VALVE"] = Surface.AORTA_VALVE.value
            if surface.name == "EPICARDIAL" and config["output_fitting"]["closed_mesh"] == True:
                mesh_data["PULMONARY_VALVE"] = Surface.PULMONARY_VALVE.value
                mesh_data["TRICUSPID_VALVE"] = Surface.TRICUSPID_VALVE.value
                mesh_data["MITRAL_VALVE"] = Surface.MITRAL_VALVE.value
                mesh_data["AORTA_VALVE"] = Surface.AORTA_VALVE.value
            meshes[surface.name] = mesh_data

    if "RV_ENDOCARDIAL" in config["output_fitting"]["output_meshes"]:
        mesh_data = {}
        mesh_data["RV_SEPTUM"] = Surface.RV_SEPTUM.value
        mesh_data["RV_FREEWALL"] = Surface.RV_FREEWALL.value
        if config["output_fitting"]["closed_mesh"]:
            mesh_data["PULMONARY_VALVE"] = Surface.PULMONARY_VALVE.value
            mesh_data["TRICUSPID_VALVE"] = Surface.TRICUSPID_VALVE.value
        meshes["RV_ENDOCARDIAL"] = mesh_data
    
    # Process each frame
    for i, frame_num in enumerate(frame_numbers):
        # Create model with smoothed control mesh
        smoothed_model = deepcopy(base_model)
        smoothed_model.update_control_mesh(smoothed_meshes[i])
        
        # Generate meshes for each surface
        for key, value in meshes.items():
            vertices = np.array([]).reshape(0, 3)
            faces_mapped = np.array([], dtype=np.int64).reshape(0, 3)
            
            offset = 0
            for type in value:
                start_fi = smoothed_model.surface_start_end[value[type]][0]
                end_fi = smoothed_model.surface_start_end[value[type]][1] + 1
                faces_et = smoothed_model.et_indices[start_fi:end_fi]
                unique_inds = np.unique(faces_et.flatten())
                vertices = np.vstack((vertices, smoothed_model.et_pos[unique_inds]))
                
                # Remap faces/indices to 0-indexing
                mapping = {old_index: new_index for new_index, old_index in enumerate(unique_inds)}
                faces_mapped = np.vstack((faces_mapped, np.vectorize(mapping.get)(faces_et) + offset))
                offset += len(smoothed_model.et_pos[unique_inds])
            
            if output_format == ".vtk":
                # Don't add suffix to filename if suffix is empty (overwriting refitted files)
                if suffix:
                    mesh_path = vtk_dir / f"{case}_{key}_{frame_num:03}{suffix}.vtk"
                else:
                    mesh_path = vtk_dir / f"{case}_{key}_{frame_num:03}.vtk"
                write_vtk_surface(str(mesh_path), vertices, faces_mapped)
            elif output_format == ".obj":
                # Don't add suffix to filename if suffix is empty (overwriting refitted files)
                if suffix:
                    mesh_path = obj_dir / f"{case}_{key}_{frame_num:03}{suffix}.obj"
                else:
                    mesh_path = obj_dir / f"{case}_{key}_{frame_num:03}.obj"
                export_to_obj(mesh_path, vertices, faces_mapped)
        
        # Generate merged mesh if requested (for smoothed refitted models only)
        if generate_merged_mesh and output_format != "none":
            # Collect vertices and faces from all three main meshes (excluding valves)
            # Store vertices and faces per surface for boundary stitching
            surface_vertices = {}
            surface_faces = {}
            surface_face_to_vertex_mapping = {}  # Track original vertex indices per surface
            vertex_offset = 0
            
            # Only include main surfaces, exclude valve surfaces
            valve_surfaces = {Surface.MITRAL_VALVE.value, Surface.AORTA_VALVE.value, 
                             Surface.PULMONARY_VALVE.value, Surface.TRICUSPID_VALVE.value}
            
            # Process each main surface mesh (LV_ENDOCARDIAL, RV_ENDOCARDIAL, EPICARDIAL)
            for key, value in meshes.items():
                if key not in ["LV_ENDOCARDIAL", "RV_ENDOCARDIAL", "EPICARDIAL"]:
                    continue
                
                mesh_vertices = np.array([]).reshape(0, 3)
                mesh_faces = np.array([], dtype=np.int64).reshape(0, 3)
                vertex_mapping = {}  # Maps original vertex index to local index (0-indexed within this surface)
                
                for type in value:
                    # Skip valve surfaces in merged mesh
                    if value[type] in valve_surfaces:
                        continue
                    
                    start_fi = smoothed_model.surface_start_end[value[type]][0]
                    end_fi = smoothed_model.surface_start_end[value[type]][1] + 1
                    faces_et = smoothed_model.et_indices[start_fi:end_fi]
                    unique_inds = np.unique(faces_et.flatten())
                    
                    # Store vertex positions and mapping (creates local 0-indexed mapping per surface)
                    for orig_idx in unique_inds:
                        if orig_idx not in vertex_mapping:
                            vertex_mapping[orig_idx] = len(mesh_vertices)
                            mesh_vertices = np.vstack((mesh_vertices, smoothed_model.et_pos[orig_idx]))
                    
                    # Remap faces/indices to local 0-indexing (within this surface)
                    local_faces = np.vectorize(vertex_mapping.get)(faces_et)
                    
                    # Flip normals for LV endocardial and RV freewall surfaces
                    # Do not flip RV septum (it's facing the correct direction)
                    should_flip = False
                    if key == "LV_ENDOCARDIAL":
                        # Flip all LV endocardial faces
                        should_flip = True
                    elif key == "RV_ENDOCARDIAL":
                        # Only flip RV freewall, not RV septum
                        if value[type] == Surface.RV_FREEWALL.value:
                            should_flip = True
                    
                    if should_flip:
                        # Reverse face winding order to flip normals
                        local_faces = np.flip(local_faces, axis=1)
                    
                    mesh_faces = np.vstack((mesh_faces, local_faces))
                
                # Store surface data
                if len(mesh_vertices) > 0:
                    surface_vertices[key] = mesh_vertices
                    surface_faces[key] = mesh_faces
                    surface_face_to_vertex_mapping[key] = (vertex_offset, vertex_offset + len(mesh_vertices))
                    vertex_offset += len(mesh_vertices)
            
            # Merge vertices at boundaries to close gaps
            # Tolerance for merging vertices (in same units as vertex coordinates)
            merge_tolerance = 0.1  # 0.1 mm or units
            
            # Combine all vertices with global indexing
            all_vertices_list = []
            vertex_surface_labels = []  # Track which surface each vertex belongs to
            global_vertex_start_idx = {}  # Track global start index for each surface
            
            global_idx = 0
            for key in ["LV_ENDOCARDIAL", "RV_ENDOCARDIAL", "EPICARDIAL"]:
                if key in surface_vertices:
                    global_vertex_start_idx[key] = global_idx
                    num_verts = len(surface_vertices[key])
                    all_vertices_list.append(surface_vertices[key])
                    vertex_surface_labels.extend([key] * num_verts)
                    global_idx += num_verts
            
            if len(all_vertices_list) > 0:
                all_vertices = np.vstack(all_vertices_list)
                vertex_surface_labels = np.array(vertex_surface_labels)
                
                # Build KD-tree for efficient nearest neighbor search
                kdtree = cKDTree(all_vertices)
                
                # Find pairs of vertices from different surfaces within tolerance
                merged_vertex_map = {}  # Maps global index to merged index
                merged_vertices = []
                used_indices = set()
                next_merged_idx = 0
                
                for i, vertex in enumerate(all_vertices):
                    if i in used_indices:
                        continue
                    
                    # Find all nearby vertices within tolerance
                    nearby_indices = kdtree.query_ball_point(vertex, merge_tolerance)
                    nearby_indices = [idx for idx in nearby_indices if idx >= i]  # Only consider current and future vertices
                    
                    # Check if any nearby vertices are from different surfaces
                    current_surface = vertex_surface_labels[i]
                    merge_candidates = [i]
                    
                    for j in nearby_indices:
                        if j == i or j in used_indices:
                            continue
                        other_surface = vertex_surface_labels[j]
                        
                        # Only merge vertices from different surfaces (boundaries)
                        # Don't merge vertices from the same surface
                        if current_surface != other_surface:
                            # Merge all boundary vertices - valve openings are already excluded
                            # since we skip valve surfaces above
                            merge_candidates.append(j)
                    
                    # Use first vertex as the merged vertex (average could be used, but this is simpler)
                    merged_vertex = all_vertices[merge_candidates[0]]
                    merged_vertices.append(merged_vertex)
                    
                    # Map all candidates to the merged index
                    for cand_idx in merge_candidates:
                        merged_vertex_map[cand_idx] = next_merged_idx
                        used_indices.add(cand_idx)
                    
                    next_merged_idx += 1
                
                # Handle vertices that weren't merged (keep them as-is)
                for i in range(len(all_vertices)):
                    if i not in used_indices:
                        merged_vertices.append(all_vertices[i])
                        merged_vertex_map[i] = next_merged_idx
                        next_merged_idx += 1
                
                # Convert to numpy array
                merged_vertices = np.array(merged_vertices)
                
                # Update face indices to use merged vertices
                merged_faces = np.array([], dtype=np.int64).reshape(0, 3)
                
                for key in ["LV_ENDOCARDIAL", "RV_ENDOCARDIAL", "EPICARDIAL"]:
                    if key in surface_faces:
                        global_start = global_vertex_start_idx[key]
                        faces = surface_faces[key]
                        
                        # Remap face indices to use merged vertex indices
                        # Faces have local indices within the surface, need to convert to global then to merged
                        remapped_faces = np.zeros_like(faces)
                        for face_idx, face in enumerate(faces):
                            for vert_idx, local_vertex_idx in enumerate(face):
                                # Convert local surface vertex index to global index
                                global_vertex_idx = local_vertex_idx + global_start
                                # Get the merged vertex index
                                if global_vertex_idx in merged_vertex_map:
                                    remapped_faces[face_idx, vert_idx] = merged_vertex_map[global_vertex_idx]
                                else:
                                    # Fallback: keep original (shouldn't happen)
                                    remapped_faces[face_idx, vert_idx] = local_vertex_idx
                        
                        merged_faces = np.vstack((merged_faces, remapped_faces))
                
                # Export merged mesh as OBJ
                if output_format == ".vtk" or output_format == ".obj":
                    # Determine output directory based on format
                    if output_format == ".vtk":
                        output_dir = vtk_dir
                    else:
                        output_dir = obj_dir
                    merged_obj_path = output_dir / f"{case}_merged_{frame_num:03}.obj"
                    export_to_obj(merged_obj_path, merged_vertices, merged_faces)
            else:
                my_logger.warning(f"No vertices found for merged mesh generation for frame {frame_num}")
        
        # Handle control mesh export if enabled
        if config["output_fitting"]["export_control_mesh"]:
            control_mesh_meshes = {}
            for surface in ControlMesh:
                control_mesh_mesh_data = {}
                if surface.name in config["output_fitting"]["output_meshes"]:
                    control_mesh_mesh_data[surface.name] = surface.value
                    if surface.name == "LV_ENDOCARDIAL" and config["output_fitting"]["closed_mesh"] == True:
                        control_mesh_mesh_data["MITRAL_VALVE"] = ControlMesh.MITRAL_VALVE.value
                        control_mesh_mesh_data["AORTA_VALVE"] = ControlMesh.AORTA_VALVE.value
                    if surface.name == "EPICARDIAL" and config["output_fitting"]["closed_mesh"] == True:
                        control_mesh_mesh_data["PULMONARY_VALVE"] = ControlMesh.PULMONARY_VALVE.value
                        control_mesh_mesh_data["TRICUSPID_VALVE"] = ControlMesh.TRICUSPID_VALVE.value
                        control_mesh_mesh_data["MITRAL_VALVE"] = ControlMesh.MITRAL_VALVE.value
                        control_mesh_mesh_data["AORTA_VALVE"] = ControlMesh.AORTA_VALVE.value
                    if surface.name == "RV_ENDOCARDIAL" and config["output_fitting"]["closed_mesh"] == True:
                        control_mesh_mesh_data["PULMONARY_VALVE"] = ControlMesh.PULMONARY_VALVE.value
                        control_mesh_mesh_data["TRICUSPID_VALVE"] = ControlMesh.TRICUSPID_VALVE.value
                    control_mesh_meshes[surface.name] = control_mesh_mesh_data
            
            for key, value in control_mesh_meshes.items():
                vertices = np.array([]).reshape(0, 3)
                faces_mapped = np.array([], dtype=np.int64).reshape(0, 3)
                
                offset = 0
                for type in value:
                    start_fi = smoothed_model.control_mesh_start_end[value[type]][0]
                    end_fi = smoothed_model.control_mesh_start_end[value[type]][1] + 1
                    faces_et = smoothed_model.et_indices_control_mesh[start_fi:end_fi]
                    unique_inds = np.unique(faces_et.flatten())
                    vertices = np.vstack((vertices, smoothed_model.control_mesh[unique_inds]))
                    
                    # Remap faces/indices to 0-indexing
                    mapping = {old_index: new_index for new_index, old_index in enumerate(unique_inds)}
                    faces_mapped = np.vstack((faces_mapped, np.vectorize(mapping.get)(faces_et) + offset))
                    offset += len(smoothed_model.control_mesh[unique_inds])
                
                if output_format == ".vtk":
                    mesh_path = vtk_dir / f"{case}_{key}_{frame_num:03}_control_mesh.vtk"
                    write_vtk_surface(str(mesh_path), vertices, faces_mapped)
                elif output_format == ".obj":
                    mesh_path = obj_dir / f"{case}_{key}_{frame_num:03}_control_mesh.obj"
                    export_to_obj(mesh_path, vertices, faces_mapped)
