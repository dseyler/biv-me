import os
import pydicom

# Update these if they don't work for your dataset
INCLUSION_TERMS = [''] # include only series that have any one of these terms in the description
EXCLUSION_TERMS = ['loc', 'molli', 't1', 't2', 'dense', 'scout', 'grid', 'flow', 'fl2d',
                   'single shot', 'report', 'document', 'segmentation', 'result', 'mapping', 'mag', 'psir', 'suiteheart',
                   'axial', 'coronal', 'transverse', 'cas', 'survey', 'nav', 'tpat', 't-pat', 'gad',
                   'cs_rt_10sl_jc', 'truefisp', 'catch', 'fl3d', 'exceptions', 'haste', 'oblique'] # exclude series with any one of these terms in the description

def extract_cines(src, dst, my_logger):
    # This function is used to preprocess the DICOM files before running the pipeline. 
    # It (hopefully) extracts only cine images and converts them to .dcm format. 

    # Create destination directory if it does not exist
    processed_dcm_dir = os.path.join(dst, 'processed-dicoms')
    os.makedirs(processed_dcm_dir, exist_ok=True) # cine .dcms will be saved here

    file_paths = []
    # Get all files in the source directory
    for root, dirs, files in os.walk(src):
        if not files:  # Skip empty directories
            continue
        for file in files:
            file_paths.append(os.path.join(root, file))

    file_names = [os.path.basename(file) for file in file_paths]
    sets = list(set(file_names))  # Get unique file names to avoid processing duplicates

    for s in sets:
        # For each set, find all files with the same name
        files = [fp for fp in file_paths if os.path.basename(fp) == s]
        for file in files:
            f = os.path.basename(file)
            root = os.path.dirname(file)
            try:
                dcm = pydicom.dcmread(os.path.join(root, f))
            except:
                my_logger.warning(f'Could not read {file}. Might not be a DICOM file.')
                continue
            
            try:
                description = dcm.SeriesDescription.lower() # lower case for easier comparison
            except:
                my_logger.warning(f'Could not find series description tag for {file}. Excluded for now.')
                continue

            if any(term in description for term in INCLUSION_TERMS) and not any(term in description for term in EXCLUSION_TERMS):
                # Save the cine images to the destination directory as .dcm files
                if not f.endswith('.dcm'):
                    f = f'{f}.dcm'  # Ensure the file has a .dcm extension
                
                if len(files) == 1: # If there's only one file with this name, just save it
                    dcm.save_as(os.path.join(processed_dcm_dir, f))
                    continue
                else:
                    idx = files.index(file) # Get the index of the file in the set
                    f = f.replace('.dcm', '')  # Remove .dcm extension for indexing (added back in next line)
                    dcm.save_as(os.path.join(processed_dcm_dir, f'{f}_{idx}.dcm'))