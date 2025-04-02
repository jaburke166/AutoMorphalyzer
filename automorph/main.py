import os
import sys
from pathlib import Path

SCRIPT_PATH = os.path.realpath(os.path.dirname(__file__))
MODULE_PATH = os.path.split(SCRIPT_PATH)[0]
PACKAGE_PATH = os.path.split(MODULE_PATH)[0]
sys.path.append(SCRIPT_PATH)
sys.path.append(MODULE_PATH)
sys.path.append(PACKAGE_PATH)

import utils
from preprocess import preprocess
from segment import segment, segment_pvbm
from measure import measure

def run(args):
    '''
    Outer function to run Automorph on a dataset.
    '''
    # Extract input/output directories
    AUTOMORPH_DATA = args['input_directory']
    AUTOMORPH_RESULTS = args['output_directory']
    AUTOMORPH_WEIGHTS = args["AutoMorph_models"]

    # Detect supported image file types
    image_list = []
    for typ in ['.bmp', 'png', '.jpg', '.jpeg']:
        image_list += list(Path(f'{AUTOMORPH_DATA}').glob(f'*{typ}'))
    image_list = [os.path.split(str(p))[1] for p in image_list]
    N = len(image_list)

    # Check to see if there are any files to process
    if N > 0:
        print(f"Found {N} to analyse.")
    else:
        print(f'Cannot find any supported files in {AUTOMORPH_DATA}. Please check directory. Exiting analysis')
        return
    
    if AUTOMORPH_WEIGHTS:
        print("\nRunning pipeline using AutoMorph's original model weights.")
    else:
        print("\nRunning pipeline using PVBM's model weights.")
    
    # Checking for model weights and downloading if so
    print('\nChecking for model weights, this may take a few minutes if model weights are not already downloaded...')
    utils.check_unpack_model_weights('binary', SCRIPT_PATH, AUTOMORPH_WEIGHTS)
    utils.check_unpack_model_weights('artery_vein', SCRIPT_PATH, AUTOMORPH_WEIGHTS)
    utils.check_unpack_model_weights('optic_disc', SCRIPT_PATH, AUTOMORPH_WEIGHTS)
    print('Model weights ready to go!')

    # Creating directory of pre-processed images
    save_path = f'{AUTOMORPH_RESULTS}/M0/images/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Running pre-processing and quality extraction
    print('\nPre-processing module.')
    preprocess.preprocess_dataset(AUTOMORPH_DATA, image_list, AUTOMORPH_RESULTS)
    print('Done!')

    # Segmentation, taking into account flag for using AutoMorph weights or PVBM model weights
    # Uses AutoMorph weights (Zhou, et al.)
    if AUTOMORPH_WEIGHTS:
        print('\nBinary vessel segmentation.')
        bin_networks = utils.get_binary_models()
        segment.binary_vessel_segmentation(bin_networks, AUTOMORPH_RESULTS)
        del bin_networks

        print('\nArtery-Vein vessel segmentation.')
        av_networks = utils.get_av_models()
        segment.arteryvein_vessel_segmentation(av_networks, AUTOMORPH_RESULTS)
        del av_networks

        print('\nOptic Disc/Cup segmentation.')
        od_networks = utils.get_od_models()
        segment.opticdisc_segmentation(od_networks, AUTOMORPH_RESULTS)
        del od_networks

    # Uses PVBM (Fhima, et al.)
    else:
        print('\nVessel segmentation.')
        segment_pvbm.vessel_segmentation(AUTOMORPH_RESULTS)

        print('\nOptic Disc segmentation.')
        segment_pvbm.opticdisc_segmentation(AUTOMORPH_RESULTS)
    print('Segmentation module complete!')

    print('\nFeature measurement module.')
    measure.feature_measurement(image_list, AUTOMORPH_RESULTS, AUTOMORPH_WEIGHTS)
    print('\nAutoMorphalyzer analysis complete!')

    # Create annotations directory for end-users to put .nii.gz files into
    # alongside a README.txt file
    text_for_readme = ['Please place your annotations in this directory. Annotations should be in .nii.gz format.',
                        'Please refer to the instructions document in AutoMorphalyzer/manual_annotations for more information.',
                        'In particular, this contains instructions on how to use ITK-Snap to create annotations, and how to save them in the correct format.',
                        'If you have any questions, please contact the developers (Jamie Burke, Jamie.Burke@ed.ac.uk).']
    if not os.path.exists(f'{AUTOMORPH_RESULTS}/annotations'):
        os.makedirs(f'{AUTOMORPH_RESULTS}/annotations')
        with open(f'{AUTOMORPH_RESULTS}/annotations/README.txt', 'w') as f:
            for line in text_for_readme:
                f.write(line + '\n')


# Once called from terminal
if __name__ == "__main__":

    print("Checking configuration file for valid inputs...")

    # Load in configuration from file
    config_path = os.path.join(MODULE_PATH, 'config.txt')
    with Path(config_path).open('r') as f:
        lines = f.readlines()
    inputs = [l.strip() for l in lines if (":" in str(l)) and ("#" not in str(l))]
    params = {p.split(": ")[0]:p.split(": ")[1] for p in inputs}

    # Make sure inputs are correct format before constructing args dict
    for key, param in params.items():

        # Checks for directory
        if "directory" in key:
            if "input" in key:
                try:
                    assert os.path.exists(param), f"The specified path:\n{param}\ndoes not exist. Check spelling or location. Exiting analysis."
                except AssertionError as msg:
                    sys.exit(msg)
            if 'output' in key:
                if not os.path.exists(param):
                    os.makedirs(param)
            continue

        # Check numerical value inputs and custom slab input
        param = param.replace(" ", "")
        if param == "":
            msg = f"No value entered for {key}. Please check config.txt. Exiting analysis"
            sys.exit(msg)
        else:
            try:
                int(param)
            except:
                msg = print(f"Value {param} for parameter {key} is not valid. Please check config.txt, Exiting analysis.")
                sys.exit(msg)

    # Construct args dict and run
    args = {key:val if ("directory" in key) else int(val) for (key,val) in params.items()}

    # run analysis
    run(args)
