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

    # Get all files in the source directory
    for root, dirs, files in os.walk(src):
        for file in files:
            try:
                dcm = pydicom.dcmread(os.path.join(root, file))
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
                if not file.endswith('.dcm'):
                    file = f'{file}.dcm'  # Ensure the file has a .dcm extension
                
                # Check if file already exists - sometimes DICOMs have non-unique names, so we need to ensure we don't overwrite them
                if os.path.exists(os.path.join(processed_dcm_dir, file)):
                    series_number = dcm.SeriesNumber
                    file = file.replace('.dcm', f'_{series_number}.dcm')  # Rename to avoid overwriting. Arbitrarily appending series number to the filename

                dcm.save_as(os.path.join(processed_dcm_dir, file))