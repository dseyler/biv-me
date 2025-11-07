import os, sys
import numpy as np
import time
import pandas as pd
import plotly.graph_objs as go
from pathlib import Path
from plotly.offline import plot
import re
import fnmatch
from copy import deepcopy
from bivme.fitting.surface_enum import Surface, ControlMesh

from bivme.fitting.BiventricularModel import BiventricularModel
from bivme.fitting.GPDataSet import GPDataSet
from bivme.fitting.surface_enum import ContourType
from bivme.fitting.diffeomorphic_fitting_utils import (
    solve_least_squares_problem,
    solve_convex_problem,
    plot_timeseries,
)
from bivme.fitting.temporal_smoothing import smooth_control_meshes_temporal
from bivme.plotting.plot_guidepoints import plot_images

from bivme.meshing.mesh_io import write_vtk_surface, export_to_obj
from loguru import logger
from rich.progress import Progress
from bivme import MODEL_RESOURCE_DIR

# This list of contours_to _plot was taken from Liandong Lee
contours_to_plot = [
    ContourType.LAX_RA,
    ContourType.LAX_LA,
    ContourType.LAX_RV_ENDOCARDIAL,
    ContourType.SAX_RV_FREEWALL,
    ContourType.LAX_RV_FREEWALL,
    ContourType.SAX_RV_SEPTUM,
    ContourType.LAX_RV_SEPTUM,
    ContourType.SAX_LV_ENDOCARDIAL,
    ContourType.SAX_LV_EPICARDIAL,
    ContourType.RV_INSERT,
    ContourType.APEX_POINT,
    ContourType.MITRAL_VALVE,
    ContourType.TRICUSPID_VALVE,
    ContourType.AORTA_VALVE,
    ContourType.PULMONARY_VALVE,
    ContourType.SAX_RV_EPICARDIAL,
    ContourType.LAX_RV_EPICARDIAL,
    ContourType.LAX_LV_ENDOCARDIAL,
    ContourType.LAX_LV_EPICARDIAL,
    ContourType.SAX_RV_OUTLET,
    ContourType.AORTA_PHANTOM,
    ContourType.TRICUSPID_PHANTOM,
    ContourType.MITRAL_PHANTOM,
    ContourType.PULMONARY_PHANTOM,
    ContourType.EXCLUDED,
]

def perform_fitting(folder: str,  config: dict, out_dir: str ="./results/", gp_suffix: str ="", si_suffix: str ="", frames_to_fit: list[int]=[], output_format: str =".vtk", my_logger: logger = logger, **kwargs) -> float:
    # performs all the BiVentricular fitting operations

    try:
        #if "iter_num" in kwargs:
        #    iter_num = kwargs.get("iter_num", None)
        #    pid = os.getpid()
        #    # print('child PID', pid)
        #    # assign a new process ID and a new CPU to the child process
        #    # iter_num corresponds to the id number of the CPU where the process will be run
        #    os.system("taskset -cp %d %d" % (iter_num, pid))

        if "id_Frame" in kwargs:
            # acquire .csv file containing patient_id, ES frame number, ED frame number if present
            case_frame_dict = kwargs.get("id_Frame", None)

        filename_info = Path(folder) / f"SliceInfoFile{si_suffix}.txt"
        if not filename_info.exists():
            my_logger.error(f"Cannot find {filename_info} file! Skipping this model")
            return -1

        # extract the patient name from the folder name
        case = os.path.basename(os.path.normpath(folder))
        my_logger.info(f"case: {case}")

        rule = re.compile(fnmatch.translate(f"GPFile_{gp_suffix}*.txt"), re.IGNORECASE)
        time_frame = [Path(folder) / Path(name) for name in os.listdir(Path(folder)) if rule.match(name)]
        frame_name = [re.search(r'GPFile_*(\d+)\.txt', str(file), re.IGNORECASE)[1] for file in time_frame]
        frame_name = sorted(frame_name)

        ed_frame = config["breathhold_correction"]["ed_frame"]
        my_logger.info(f'ED set to frame #{config["breathhold_correction"]["ed_frame"]}')

        if len(frames_to_fit) == 0:
            frames_to_fit = np.unique(
                frame_name
            )  # if you want to fit all _frames#

        # create a separate output folder for each patient
        output_folder = Path(out_dir) / case
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        # create log files where to store fitting errors and shift
        shift_file = output_folder / f"shift_file{gp_suffix}.txt"
        pos_file = output_folder / f"pos_file{gp_suffix}.txt"

        # The next lines are used to measure shift using only a key frame
        if config["breathhold_correction"]["shifting"] == "derived_from_ed":
            my_logger.info("Shift measured only at ED frame")
            filename = Path(folder) / f"GPFile_{gp_suffix}{frame_name[ed_frame]:03}.txt"

            if not filename.exists():
                my_logger.error(f"Cannot find {filename} file! Skipping this model")
                return -1

            ed_dataset = GPDataSet(
                str(filename),
                str(filename_info),
                case,
                sampling=config["gp_processing"]["sampling"],
                time_frame_number=ed_frame,
            )
            if not ed_dataset.success:
                return -1
            
            # Get slice IDs that should use only LV contours for shift calculation
            lv_only_slice_ids = []
            if config.get("gp_processing", {}).get("filter_sax_lv_epicardial", False):
                lv_only_slice_ids = config.get("gp_processing", {}).get("filter_sax_lv_epicardial_slice_ids", [])
                if not isinstance(lv_only_slice_ids, list):
                    lv_only_slice_ids = []
            
            result_at_ed = ed_dataset.sinclair_slice_shifting(my_logger, lv_only_slice_ids=lv_only_slice_ids)
            _, _ = ed_dataset.get_unintersected_slices()

            ##TODO remove basal slice (maybe looking at the distance between the contours centroid and the projection of the line mitral centroid/apex)
            shift_to_apply = result_at_ed[0]
            updated_slice_position = result_at_ed[1]

            with shift_file.open("w", encoding ="utf-8") as file:
                file.write("shift measured only at ED: frame " + str(ed_frame) + "\n")
                file.write(str(shift_to_apply))
                file.close()

            with pos_file.open("w", encoding ="utf-8") as file:
                file.write("pos measured only at ED: frame " + str(ed_frame) + "\n")
                file.write(str(updated_slice_position))
                file.close()

        elif config["breathhold_correction"]["shifting"] == "average_all_frames":
            my_logger.info("Shift measured on all the frames and averaged")
            shift_to_apply = 0  # 2D translation
            updated_slice_position = 0
            counter = 0
            
            # Get slice IDs that should use only LV contours for shift calculation
            lv_only_slice_ids = []
            if config.get("gp_processing", {}).get("filter_sax_lv_epicardial", False):
                lv_only_slice_ids = config.get("gp_processing", {}).get("filter_sax_lv_epicardial_slice_ids", [])
                if not isinstance(lv_only_slice_ids, list):
                    lv_only_slice_ids = []
            
            for frame in sorted(frames_to_fit):
                num = int(frame)
                filename = Path(folder) / f"GPFile_{gp_suffix}{num:03}.txt"
                if not filename.exists():
                    my_logger.error(f"Cannot find {filename} file! Skipping this model")
                    return -1

                dataset = GPDataSet(
                    str(filename),
                    str(filename_info),
                    case,
                    sampling=config["gp_processing"]["sampling"],
                    time_frame_number=num,
                )

                if frame == frames_to_fit[ed_frame]:
                    ed_dataset = deepcopy(dataset)
                if not dataset.success:
                    continue
                result_at_t = dataset.sinclair_slice_shifting(my_logger, lv_only_slice_ids=lv_only_slice_ids)

                shift_to_apply += result_at_t[0]
                updated_slice_position += result_at_t[1]
                counter += 1

            shift_to_apply /= counter
            updated_slice_position /= counter

            with shift_file.open("w", encoding ="utf-8") as file:
                file.write("Average shift \n")
                file.write(str(shift_to_apply))
                file.close()

            with pos_file.open("w", encoding ="utf-8") as file:
                file.write("Average shift \n")
                file.write(str(updated_slice_position))
                file.close()

        elif config["breathhold_correction"]["shifting"] == "none":
            my_logger.info("No correction applied")
            filename = Path(folder) / f"GPFile_{gp_suffix}{frame_name[ed_frame]:03}.txt"
            if not filename.exists():
                my_logger.error(f"Cannot find {filename} file! Skipping this model")
                return -1

            ed_dataset = GPDataSet(
                str(filename),
                str(filename_info),
                case,
                sampling=config["gp_processing"]["sampling"],
                time_frame_number=ed_frame,
            )
            if not ed_dataset.success:
                return -1

        else:
            my_logger.error(f'Method {config["breathhold_correction"]["shifting"]} unavailable.  Allowed values: none, derived_from_ed or average_all_frame. No correction applied')
            return -1

        biventricular_model = BiventricularModel(MODEL_RESOURCE_DIR, case)

        my_logger.info(f"Calculating pose and scale {str(case)}...")
        biventricular_model.update_pose_and_scale(ed_dataset)
        aligned_biventricular_model = deepcopy(biventricular_model)

        # initialise time series lists
        my_logger.info(f"Fitting of {str(case)}")

        residuals = 0
        frame_residuals = {}  # Track residuals per frame for temporal smoothing
        with Progress(auto_refresh=False, transient=True) as progress:
            task = progress.add_task(f"Processing {len(frames_to_fit)} frames of {case}", total=len(frames_to_fit))
            console = progress

            image_grids = {}

            for idx, num in enumerate(sorted(frames_to_fit)):
                progress.refresh()
                num = int(num)  # frame number

                my_logger.info(f"Processing frame #{num}")
                model_file = Path(
                    output_folder, f"{case}{gp_suffix}_model_frame_{num:03}.txt"
                )
                model_file.touch(exist_ok=True)

                filename = Path(folder) / f"GPFile_{gp_suffix}{num:03}.txt"
                if not filename.exists():
                    my_logger.error(f"Cannot find {filename} file! Skipping this model")
                    continue

                data_set = GPDataSet(
                    str(filename), str(filename_info), case, sampling=config["gp_processing"]["sampling"], time_frame_number=num
                )
                if not data_set.success:
                    my_logger.error(f"Cannot initialize GPDataSet! Skipping this frame")
                    continue

                if config["breathhold_correction"]["shifting"] != "none":
                    data_set.apply_slice_shift(shift_to_apply, updated_slice_position)
                    data_set.get_unintersected_slices()

                # Filter SAX_LV_EPICARDIAL guidepoints (if enabled) - after breathhold correction
                if config.get("gp_processing", {}).get("filter_sax_lv_epicardial", False):
                    slice_ids_to_filter = config.get("gp_processing", {}).get("filter_sax_lv_epicardial_slice_ids", [17, 18])
                    data_set.filter_sax_lv_epicardial_points(slice_ids_to_filter, my_logger)

                # Generates RV epicardial point if they have not been contoured
                if sum(data_set.contour_type == (ContourType.SAX_RV_EPICARDIAL)) == 0 and sum(data_set.contour_type == ContourType.LAX_RV_EPICARDIAL) == 0:
                    my_logger.info('Generating RV epicardial points')
                    _, _, _ = data_set.create_rv_epicardium(
                        rv_thickness=3
                    )

                try:
                    _ = data_set.create_valve_phantom_points(config["gp_processing"]["num_of_phantom_points_mv"], ContourType.MITRAL_VALVE)
                except:
                    my_logger.warning('Error in creating mitral phantom points')

                try:
                    _ = data_set.create_valve_phantom_points(config["gp_processing"]["num_of_phantom_points_tv"], ContourType.TRICUSPID_VALVE)
                except:
                    my_logger.warning('Error in creating tricuspid phantom points')

                try:
                    _ = data_set.create_valve_phantom_points(config["gp_processing"]["num_of_phantom_points_pv"], ContourType.PULMONARY_VALVE)
                except:
                    my_logger.warning('Error in creating pulmonary phantom points')

                try:
                    _ = data_set.create_valve_phantom_points(config["gp_processing"]["num_of_phantom_points_av"], ContourType.AORTA_VALVE)
                except:
                    my_logger.warning('Error in creating aorta phantom points')

                # Example on how to set different weights for different points group (R.B.)
                data_set.weights[data_set.contour_type == ContourType.MITRAL_PHANTOM] = 1
                data_set.weights[data_set.contour_type == ContourType.AORTA_PHANTOM] = 1
                data_set.weights[data_set.contour_type == ContourType.PULMONARY_PHANTOM] = 1
                data_set.weights[data_set.contour_type == ContourType.TRICUSPID_PHANTOM] = 1

                data_set.weights[data_set.contour_type == ContourType.APEX_POINT] = 1
                data_set.weights[data_set.contour_type == ContourType.RV_INSERT] = 2

                data_set.weights[data_set.contour_type == ContourType.MITRAL_VALVE] = 1
                data_set.weights[data_set.contour_type == ContourType.AORTA_VALVE] = 1
                data_set.weights[data_set.contour_type == ContourType.PULMONARY_VALVE] = 1
                data_set.weights[data_set.contour_type == ContourType.TRICUSPID_VALVE] = 1

                # Perform linear fit
                biventricular_model = deepcopy(aligned_biventricular_model)
                pulmonary_circularity_weight = config["fitting_weights"].get("pulmonary_artery_circularity", 0.0)
                solve_least_squares_problem(
                    biventricular_model, 
                    config["fitting_weights"]["guide_points"], 
                    data_set, 
                    my_logger,
                    pulmonary_circularity_weight=pulmonary_circularity_weight
                )

                ## Perform diffeomorphic fit
                frame_residual = solve_convex_problem(
                    biventricular_model,
                    data_set,
                    config["fitting_weights"]["guide_points"],
                    config["fitting_weights"]["convex_problem"],
                    config["fitting_weights"]["transmural"],
                    my_logger,
                    pulmonary_circularity_weight=pulmonary_circularity_weight,
                )
                frame_residuals[num] = frame_residual  # Store per-frame residual
                residuals += frame_residual / len(sorted(frames_to_fit))

                # Pulmonary artery refitting (if enabled)
                if config.get("pulmonary_artery_refitting", {}).get("enabled", False):
                    my_logger.info(f"Performing pulmonary artery refitting for frame #{num}")
                    
                    # Generate new longitudinal-aligned pulmonary phantom points
                    num_pv_points = config["gp_processing"]["num_of_phantom_points_pv"]
                    new_phantom_points = GPDataSet.generate_longitudinal_aligned_pulmonary_phantom_points(
                        biventricular_model, num_pv_points, my_logger
                    )
                    
                    if len(new_phantom_points) > 0:
                        # Create a deepcopy of the dataset for refitting
                        data_set_refit = deepcopy(data_set)
                        
                        # Replace pulmonary valve points with new phantom points
                        data_set_refit.replace_pulmonary_valve_points(new_phantom_points, weight=1.0)
                        
                        # Reinitialize landmarks for refit dataset
                        data_set_refit.identify_mitral_valve_points()
                        
                        # Create new model instances for refitting
                        biventricular_model_refit = deepcopy(aligned_biventricular_model)
                        
                        # Perform refitting (least squares + diffeomorphic fit)
                        # Use same weights but disable circularity regularization during refitting
                        solve_least_squares_problem(
                            biventricular_model_refit,
                            config["fitting_weights"]["guide_points"],
                            data_set_refit,
                            my_logger,
                            pulmonary_circularity_weight=0.0  # Disabled during refitting
                        )
                        
                        solve_convex_problem(
                            biventricular_model_refit,
                            data_set_refit,
                            config["fitting_weights"]["guide_points"],
                            config["fitting_weights"]["convex_problem"],
                            config["fitting_weights"]["transmural"],
                            my_logger,
                            pulmonary_circularity_weight=0.0  # Disabled during refitting
                        )
                        
                        # Save refitted model .txt file
                        refitted_output_folder = Path(output_folder, "refitted")
                        refitted_output_folder.mkdir(exist_ok=True)
                        refitted_model_file = Path(
                            refitted_output_folder, f"{case}{gp_suffix}_model_frame_{num:03}.txt"
                        )
                        refitted_model_data = {
                            "x": biventricular_model_refit.control_mesh[:, 0],
                            "y": biventricular_model_refit.control_mesh[:, 1],
                            "z": biventricular_model_refit.control_mesh[:, 2],
                            "Frame": [num] * len(biventricular_model_refit.control_mesh[:, 2]),
                        }
                        refitted_model_data_frame = pd.DataFrame(data=refitted_model_data)
                        with open(refitted_model_file, "w") as file:
                            file.write(
                                refitted_model_data_frame.to_csv(
                                    header=True, index=False, sep=",", lineterminator="\n"
                                )
                            )
                        
                        # Generate HTML plot for refitted model
                        if config["plotting"]["generate_plots_fitting"]:
                            refitted_contour_plots = data_set_refit.plot_dataset(contours_to_plot)
                            refitted_model = biventricular_model_refit.plot_surface(
                                "rgb(0,127,0)", "rgb(0,127,127)", "rgb(127,0,0)", "all"
                            )
                            refitted_data = refitted_contour_plots + refitted_model
                            
                            if config["plotting"]["include_images"]:
                                image_dir = Path(config["input_pp"]["processing"]) / config["input_pp"]["batch_ID"] / os.path.basename(case) / "images"
                                if image_dir.exists():
                                    output_dir = Path(output_folder) / "images" if config["plotting"]["export_images"] else None
                                    refitted_image_plots, _ = plot_images(image_dir, data_set_refit, {}, output_dir, shift_to_apply, num)
                                    refitted_data += refitted_image_plots
                            
                            output_folder_html_refitted = Path(output_folder, "html_refitted")
                            output_folder_html_refitted.mkdir(exist_ok=True)
                            
                            refitted_figure = go.Figure(data=refitted_data)
                            refitted_figure.update_layout(
                                paper_bgcolor='white',
                                title=f"Refitted model and guidepoints for {case} - Frame {num:03}",
                            )
                            refitted_figure.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
                            
                            plot(
                                refitted_figure,
                                filename=os.path.join(
                                    output_folder_html_refitted, f"{case}_refitted_model_frame_{num:03}.html"
                                ),
                                auto_open=False,
                            )
                        
                        # Generate VTK/OBJ files for refitted model
                        if output_format != "none":
                            refitted_meshes = {}
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
                                    refitted_meshes[surface.name] = mesh_data
                            
                            if "RV_ENDOCARDIAL" in config["output_fitting"]["output_meshes"]:
                                rv_mesh_data = {}
                                rv_mesh_data["RV_SEPTUM"] = Surface.RV_SEPTUM.value
                                rv_mesh_data["RV_FREEWALL"] = Surface.RV_FREEWALL.value
                                if config["output_fitting"]["closed_mesh"]:
                                    rv_mesh_data["PULMONARY_VALVE"] = Surface.PULMONARY_VALVE.value
                                    rv_mesh_data["TRICUSPID_VALVE"] = Surface.TRICUSPID_VALVE.value
                                refitted_meshes["RV_ENDOCARDIAL"] = rv_mesh_data
                            
                            # Create vtk_PA output directory
                            output_folder_vtk_pa = Path(output_folder, "vtk_PA")
                            output_folder_vtk_pa.mkdir(exist_ok=True)
                            
                            for key, value in refitted_meshes.items():
                                vertices = np.array([]).reshape(0, 3)
                                faces_mapped = np.array([], dtype=np.int64).reshape(0, 3)
                                
                                offset = 0
                                for type in value:
                                    start_fi = biventricular_model_refit.surface_start_end[value[type]][0]
                                    end_fi = biventricular_model_refit.surface_start_end[value[type]][1] + 1
                                    faces_et = biventricular_model_refit.et_indices[start_fi:end_fi]
                                    unique_inds = np.unique(faces_et.flatten())
                                    vertices = np.vstack((vertices, biventricular_model_refit.et_pos[unique_inds]))
                                    
                                    # remap faces/indices to 0-indexing
                                    mapping = {old_index: new_index for new_index, old_index in enumerate(unique_inds)}
                                    faces_mapped = np.vstack((faces_mapped, np.vectorize(mapping.get)(faces_et) + offset))
                                    offset += len(biventricular_model_refit.et_pos[unique_inds])
                                
                                if output_format == ".vtk":
                                    mesh_path = Path(
                                        output_folder_vtk_pa, f"{case}_{key}_{num:03}.vtk"
                                    )
                                    write_vtk_surface(str(mesh_path), vertices, faces_mapped)
                                    my_logger.success(f"{case}_{key}_{num:03}.vtk successfully saved to {output_folder_vtk_pa}")
                                
                                elif output_format == ".obj":
                                    mesh_path = Path(
                                        output_folder_vtk_pa, f"{case}_{key}_{num:03}.obj"
                                    )
                                    export_to_obj(mesh_path, vertices, faces_mapped)
                                    my_logger.success(f"{case}_{key}_{num:03}.obj successfully saved to {output_folder_vtk_pa}")
                    else:
                        my_logger.warning(f"Could not generate pulmonary phantom points for frame #{num}, skipping refitting")

                if config["plotting"]["generate_plots_fitting"]:
                    # Plot final results
                    contour_plots = data_set.plot_dataset(contours_to_plot)
                    model = biventricular_model.plot_surface(
                        "rgb(0,127,0)", "rgb(0,127,127)", "rgb(127,0,0)", "all"
                    )

                    data = contour_plots + model

                    if config["plotting"]["include_images"]:
                        image_dir = Path(config["input_pp"]["processing"]) / config["input_pp"]["batch_ID"] / os.path.basename(case) / "images"

                        if config["plotting"]["export_images"]:
                            output_dir = Path(output_folder) / "images"
                            output_dir.mkdir(exist_ok=True)
                        else:
                            output_dir = None

                        if not image_dir.exists():
                            my_logger.warning(f"Image directory {image_dir} does not exist! Skipping image plotting")

                        else:
                            image_plots, image_grids = plot_images(image_dir, data_set, image_grids, output_dir, shift_to_apply, num)
                            data += image_plots
                        
                    output_folder_html = Path(output_folder, f"html{gp_suffix}")
                    output_folder_html.mkdir(exist_ok=True)

                    figure = go.Figure(data=data)

                    figure.update_layout(
                        paper_bgcolor='white',                
                        title=f"Model and guidepoints for {case} - Frame {num:03}",
                    )
                    figure.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False)

                    plot(
                        figure,
                        filename=os.path.join(
                            output_folder_html, f"{case}_fitted_model_frame_{num:03}.html"
                        ),
                        auto_open=False,
                    )

                # save results in .txt format, one file for each frame
                model_data = {
                    "x": biventricular_model.control_mesh[:, 0],
                    "y": biventricular_model.control_mesh[:, 1],
                    "z": biventricular_model.control_mesh[:, 2],
                    "Frame": [num] * len(biventricular_model.control_mesh[:, 2]),
                }

                model_data_frame = pd.DataFrame(data=model_data)
                with open(model_file, "w") as file:
                    file.write(
                        model_data_frame.to_csv(
                            header=True, index=False, sep=",", lineterminator="\n"
                        )
                    )

                if output_format != "none":
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
                        mesh_data["RV_SEPTUM"] = Surface.RV_SEPTUM.value
                        mesh_data["RV_FREEWALL"] = Surface.RV_FREEWALL.value
                        if config["output_fitting"]["closed_mesh"]:
                            mesh_data["PULMONARY_VALVE"] = Surface.PULMONARY_VALVE.value
                            mesh_data["TRICUSPID_VALVE"] = Surface.TRICUSPID_VALVE.value
                        meshes["RV_ENDOCARDIAL"] = mesh_data

                    ##TODO remove duplicated code here - not sure how yet
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

                    for key, value in meshes.items():
                        vertices = np.array([]).reshape(0, 3)
                        faces_mapped = np.array([], dtype=np.int64).reshape(0, 3)

                        offset = 0
                        for type in value:
                            start_fi = biventricular_model.surface_start_end[value[type]][0]
                            end_fi = biventricular_model.surface_start_end[value[type]][1] + 1
                            faces_et = biventricular_model.et_indices[start_fi:end_fi]
                            unique_inds = np.unique(faces_et.flatten())
                            vertices = np.vstack((vertices, biventricular_model.et_pos[unique_inds]))

                            # remap faces/indices to 0-indexing
                            mapping = {old_index: new_index for new_index, old_index in enumerate(unique_inds)}
                            faces_mapped = np.vstack((faces_mapped, np.vectorize(mapping.get)(faces_et) + offset))
                            offset += len(biventricular_model.et_pos[unique_inds])

                        if output_format == ".vtk":
                            output_folder_vtk = Path(output_folder, f"vtk{gp_suffix}")
                            output_folder_vtk.mkdir(exist_ok=True)
                            mesh_path = Path(
                                output_folder_vtk, f"{case}_{key}_{num:03}.vtk"
                            )
                            write_vtk_surface(str(mesh_path), vertices, faces_mapped)
                            my_logger.success(f"{case}_{key}_{num:03}.vtk successfully saved to {output_folder_vtk}")

                        elif output_format == ".obj":
                            output_folder_obj = Path(output_folder, f"obj{gp_suffix}")
                            output_folder_obj.mkdir(exist_ok=True)
                            mesh_path = Path(
                                output_folder_obj, f"{case}_{key}_{num:03}.obj"
                            )
                            export_to_obj(mesh_path, vertices, faces_mapped)
                            my_logger.success(f"{case}_{key}_{num:03}.obj successfully saved to {output_folder_obj}")
                        else:
                            my_logger.error('argument format must be .obj or .vtk')
                            return -1

                    ##TODO remove duplicated code here - not sure how yet
                    if config["output_fitting"]["export_control_mesh"]:
                        for key, value in control_mesh_meshes.items():
                            vertices = np.array([]).reshape(0, 3)
                            faces_mapped = np.array([], dtype=np.int64).reshape(0, 3)

                            offset = 0
                            for type in value:
                                start_fi = biventricular_model.control_mesh_start_end[value[type]][0]
                                end_fi = biventricular_model.control_mesh_start_end[value[type]][1] + 1
                                faces_et = biventricular_model.et_indices_control_mesh[start_fi:end_fi]
                                unique_inds = np.unique(faces_et.flatten())
                                vertices = np.vstack((vertices, biventricular_model.control_mesh[unique_inds]))

                                # remap faces/indices to 0-indexing
                                mapping = {old_index: new_index for new_index, old_index in enumerate(unique_inds)}
                                faces_mapped = np.vstack((faces_mapped, np.vectorize(mapping.get)(faces_et) + offset))
                                offset += len(biventricular_model.control_mesh[unique_inds])

                            if output_format == ".vtk":
                                output_folder_vtk = Path(output_folder, f"vtk{gp_suffix}")
                                output_folder_vtk.mkdir(exist_ok=True)
                                mesh_path = Path(
                                    output_folder_vtk, f"{case}_{key}_{num:03}_control_mesh.vtk"
                                )
                                write_vtk_surface(str(mesh_path), vertices, faces_mapped)
                                my_logger.success(f"{case}_{key}_{num:03}_control_mesh.vtk successfully saved to {output_folder_vtk}")

                            elif output_format == ".obj":
                                output_folder_obj = Path(output_folder, f"obj{gp_suffix}")
                                output_folder_obj.mkdir(exist_ok=True)
                                mesh_path = Path(
                                    output_folder_obj, f"{case}_{key}_{num:03}_control_mesh.obj"
                                )
                                export_to_obj(mesh_path, vertices, faces_mapped)
                                my_logger.success(f"{case}_{key}_{num:03}_control_mesh.obj successfully saved to {output_folder_obj}")
                            else:
                                my_logger.error('argument format must be .obj or .vtk')
                                return -1

                progress.advance(task)
        
        # Apply temporal smoothing if enabled
        if config.get("temporal_smoothing", {}).get("enabled", False):
            # If refitting is enabled, smooth refitted models instead of original
            if config.get("pulmonary_artery_refitting", {}).get("enabled", False):
                my_logger.info("Applying temporal smoothing to refitted models...")
                # Use refitted directory for input
                refitted_folder = Path(output_folder, "refitted")
                if refitted_folder.exists():
                    smooth_control_meshes_temporal(
                        refitted_folder, case, gp_suffix, folder, filename_info, si_suffix, 
                        config, output_format, my_logger, vtk_output_subdir="vtk_PA"
                    )
                else:
                    my_logger.warning("Refitted folder not found, skipping temporal smoothing")
            else:
                my_logger.info("Applying temporal smoothing to fitted models...")
                smooth_control_meshes_temporal(
                    output_folder, case, gp_suffix, folder, filename_info, si_suffix, 
                    config, output_format, my_logger
                )
            my_logger.success("Temporal smoothing completed.")
        
        return residuals
    except KeyboardInterrupt:
        return -1
    
def validate_config(config, mylogger):
    assert Path(config["input_fitting"]["gp_directory"]).exists(), \
        f'gp_directory does not exist. Cannot find {config["input_fitting"]["gp_directory"]}!'

    if not (config["breathhold_correction"]["shifting"] == "derived_from_ed" or config["breathhold_correction"]["shifting"] == "average_all_frames" or config["breathhold_correction"]["shifting"] == "none"):
        mylogger.error(f'argument shifting must be derived_from_ed, average_all_frames or none. {config["breathhold_correction"]["shifting"]} given.')
        sys.exit(0)

    if not (config["output_fitting"]["mesh_format"].endswith('.obj') or config["output_fitting"]["mesh_format"].endswith('.vtk') or config["output_fitting"]["mesh_format"] == 'none'):
        mylogger.error(f'argument mesh_format must be .obj, .vtk or none. {config["output_fitting"]["mesh_format"]} given.')
        sys.exit(0)

    for mesh in config["output_fitting"]["output_meshes"]:
        if mesh not in ["LV_ENDOCARDIAL", "RV_ENDOCARDIAL", "EPICARDIAL"]:
            mylogger.error(f'argument output_meshes invalid. {mesh} given. Allowed values are "LV_ENDOCARDIAL", "RV_ENDOCARDIAL", "EPICARDIAL"')
            sys.exit(0)

    if not (config["output_fitting"]["mesh_format"].endswith('.obj') or config["output_fitting"]["mesh_format"].endswith('.vtk') or config["output_fitting"]["mesh_format"] == 'none'):
        mylogger.error(f'argument mesh_format must be .obj, .vtk or none. {config["output_fitting"]["mesh_format"]} given.')
        sys.exit(0)

    if not (config["output_fitting"]["closed_mesh"] == True or config["output_fitting"]["closed_mesh"] == False):
        mylogger.error(f'argument closed_mesh must be true or false. {config["output_fitting"]["closed_mesh"]} given.')
        sys.exit(0)

    if not (config["output_fitting"]["overwrite"] == True or config["output_fitting"]["overwrite"] == False):
        mylogger.error(f'argument overwrite must be true or false. {config["output_fitting"]["overwrite"]} given.')
        sys.exit(0)
