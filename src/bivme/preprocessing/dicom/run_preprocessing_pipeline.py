import os
from pathlib import Path
import sys
import torch
import time
import shutil
import datetime
from loguru import logger

import warnings
warnings.filterwarnings('ignore')

# Import modules
from bivme.preprocessing.dicom.extract_cines import extract_cines
from bivme.preprocessing.dicom.select_views import select_views
from bivme.preprocessing.dicom.segment_views import segment_views
from bivme.preprocessing.dicom.correct_phase_mismatch import correct_phase_mismatch
from bivme.preprocessing.dicom.generate_contours import generate_contours
from bivme.preprocessing.dicom.export_guidepoints import export_guidepoints
from bivme.plotting.plot_guidepoints import generate_html # for plotting guidepoints


def load_precomputed_segmentations(dst, mylogger):
    """
    Load precomputed segmentations from the processing directory.
    This function assumes segmentations are already available in the expected format.
    """
    segmentation_original = '/Users/dseyler/Documents/Marsden_Lab/cine_segmentation/Manual_segmentions/segmentations_newest'
    segmentation_dir = os.path.join(dst, 'segmentations')
    # Create segmentation directory if it doesn't exist
    os.makedirs(segmentation_dir, exist_ok=True)

    # Copy original segmentations to segmentation_dir
    if os.path.exists(segmentation_original):
        mylogger.info(f'Copying segmentations from {segmentation_original} to {segmentation_dir}')
        # Copy all contents including subdirectories
        shutil.copytree(segmentation_original, segmentation_dir, dirs_exist_ok=True)
    else:
        mylogger.warning(f'Original segmentation directory not found at {segmentation_original}')
    
    if not os.path.exists(segmentation_dir):
        mylogger.error(f'Segmentation directory not found at {segmentation_dir}. Please ensure precomputed segmentations are available or set use_precomputed_segmentations=false.')
        raise FileNotFoundError(f'Precomputed segmentations not found at {segmentation_dir}. Please ensure segmentations are available or set use_precomputed_segmentations=false.')
    else:
        mylogger.info(f'Found existing segmentation directory at {segmentation_dir}')
        
        # Verify that all expected view directories exist
        views = ['SAX', '2ch', '3ch', '4ch', 'RVOT']
        for view in views:
            view_dir = os.path.join(segmentation_dir, view)
            if os.path.exists(view_dir):
                num_files = len([f for f in os.listdir(view_dir) if f.endswith('.nii.gz')])
                mylogger.info(f'Found {num_files} segmentation files for {view} view')
            else:
                mylogger.warning(f'No segmentation directory found for {view} view')
                os.makedirs(view_dir, exist_ok=True)


def perform_preprocessing(case, config, mylogger):
    # Path: src/bivme/preprocessing/dicom/models
    MODEL_DIR = Path(os.path.dirname(__file__)) / 'models'

    # Unpack config parameters
    # Input
    src = os.path.join(config["input_pp"]["source"], case)

    # Processing
    dst = os.path.join(config["input_pp"]["processing"], config["input_pp"]["batch_ID"])
    dst = os.path.join(dst, case) # destination directory for processed files
    if os.path.exists(dst):
        shutil.rmtree(dst) # remove existing directory
    os.makedirs(dst, exist_ok=True) # create new directory
    
    states = os.path.join(config["input_pp"]["states"], config["input_pp"]["batch_ID"])
    states = os.path.join(states, case, config["input_pp"]["analyst_id"]) # destination directory for view predictions which don't get overwritten, and log files
    os.makedirs(states, exist_ok=True)

    # Output
    output = os.path.join(config["output_pp"]["output_directory"], config["input_pp"]["batch_ID"])
    output = os.path.join(output, case) # output directory for guidepoints
    if os.path.exists(output):
        shutil.rmtree(output) # remove existing directory
    os.makedirs(output, exist_ok=True) # create new directory

    plotting = os.path.join(config["input_pp"]["processing"], config["input_pp"]["batch_ID"]) # save the plotted htmls in processed directory

    # Logging
    if not config["logging"]["show_detailed_logging"]:
        mylogger.remove()

    if config["logging"]["generate_log_file"]:
        log_level = "DEBUG"
        log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <level>{level: <8}</level> | <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>"
        time_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        logger_id = mylogger.add(f'{output}/log_file_{time_string}.log', level=log_level, format=log_format,
                    colorize=False, backtrace=True,
                    diagnose=True)

    # Check if GPU is available (torch)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        mylogger.warning('No GPU available. Using CPU instead. This may be very slow!')

    mylogger.info(f'Using device: {device}')

    ## Step 0: Pre-preprocessing (separate cines from non-cines)
    mylogger.info(f'Finding cines...')
    extract_cines(src, dst, mylogger)

    src = os.path.join(dst, 'processed-dicoms') # Update source directory
    mylogger.success(f'Pre-preprocessing complete. Cines extracted to {src}.')

    ## Step 1: View selection
    slice_info_df, num_phases = select_views(case, src, dst, MODEL_DIR, states, config["view-selection"]["option"], 
                                                            config["view-selection"]["correct_mode"], mylogger)

    mylogger.success(f'View selection complete.')
    mylogger.info(f'Number of phases: {num_phases}')

    ## Step 2: Segmentation
    #if config["view-selection"]["use_precomputed_segmentations"]:
    mylogger.info(f'Using precomputed segmentations...')
    # Load existing segmentations from processing directory
    segment_views(dst, MODEL_DIR, slice_info_df, mylogger)
    load_precomputed_segmentations(dst, mylogger)
    mylogger.success(f'Precomputed segmentations loaded successfully.')
    #else:
    #    seg_start_time = time.time()
    #    mylogger.info(f'Starting segmentation...')
    #    segment_views(dst, MODEL_DIR, slice_info_df, mylogger) # TODO: Find a way to suppress nnUnet output
    #    seg_end_time = time.time()
    #    mylogger.success(f'Segmentation complete. Time taken: {seg_end_time-seg_start_time} seconds.')

    ## Step 2.1: Correct phase mismatch (if required)
    correct_phase_mismatch(dst, slice_info_df, num_phases, mylogger) 

    ## Step 3: Guide point extraction
    slice_dict = generate_contours(dst, slice_info_df, num_phases, mylogger)
    mylogger.success(f'Guide points generated successfully.')

    ## Step 4: Export guide points
    export_guidepoints(dst, output, slice_dict, config["contouring"]["smooth_landmarks"])
    mylogger.success(f'Guide points exported successfully.')

    ## Step 5: Generate HTML (optional) of guide points for visualisation
    if config["plotting"]["generate_plots_preprocessing"]:
        if config["plotting"]["include_images"]:
            image_path = os.path.join(dst, 'images')  # Path to the image file for plotting
        else:
            image_path = None

        generate_html(output, out_dir=plotting, gp_suffix='', si_suffix='', frames_to_fit=[], my_logger=mylogger, model_path=None, image_path=image_path)

        mylogger.success(f'Guidepoints plotted at {os.path.join(plotting,case,"html")}.')

    if config["logging"]["generate_log_file"]:
        mylogger.remove(logger_id)
        # Copy log file to states directory
        shutil.copyfile(f'{output}/log_file_{time_string}.log', os.path.join(states, f'log_file_{time_string}.log'))

def validate_config(config, mylogger):
    assert os.path.exists(config["input_pp"]["source"]), \
        f'DICOM folder does not exist! Make sure to add the correct directory under "source" in the config file.'
    
    if not (config["view-selection"]["option"] == "default" or config["view-selection"]["option"] == "image-only"  or config["view-selection"]["option"] == "load"):
        mylogger.error(f'Invalid view selection option: {config["view-selection"]["option"]}. Must be "default", "image-only", or "load".')
        sys.exit(0)

    if not (config["view-selection"]["correct_mode"] == "automatic" or config["view-selection"]["correct_mode"] == "adaptive" or config["view-selection"]["correct_mode"] == "manual"):
        mylogger.error(f'Invalid correct mode: {config["view-selection"]["correct_mode"]}. Must be "automatic", "adaptive", or "manual".')
        sys.exit(0)

    if not (config["contouring"]["smooth_landmarks"] == True or config["contouring"]["smooth_landmarks"] == False):
        mylogger.error(f'Invalid smooth_landmarks option: {config["contouring"]["smooth_landmarks"]}. Must be true or false.')
        sys.exit(0)

    if not (config["output_pp"]["overwrite"] == True or config["output_pp"]["overwrite"] == False):
        mylogger.error(f'Invalid overwrite option: {config["output_pp"]["overwrite"]}. Must be true or false.')
        sys.exit(0)

    if not (config["view-selection"]["use_precomputed_segmentations"] == True or config["view-selection"]["use_precomputed_segmentations"] == False):
        mylogger.error(f'Invalid use_precomputed_segmentations option: {config["view-selection"]["use_precomputed_segmentations"]}. Must be true or false.')
        sys.exit(0)
