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
from segment import segment
from measure import measure

def run(args):
    '''
    Outer function to run Automorph on a dataset.
    '''
    # Extract input/output directories
    AUTOMORPH_DATA = args['input_directory']
    AUTOMORPH_RESULTS = args['output_directory']

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

    # Creating directory of pre-processed images
    save_path = f'{AUTOMORPH_RESULTS}/M0/images/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Running pre-processing and quality extraction
    print('\nPre-processing module.')
    preprocess.preprocess_dataset(AUTOMORPH_DATA, 
                                   image_list, 
                                   AUTOMORPH_RESULTS)
    print('Done!')

    # Load binary segmentation models
    print('\nBinary vessel segmentation.')
    bin_networks = utils.get_binary_models()
    segment.binary_vessel_segmentation(bin_networks,
                                       AUTOMORPH_RESULTS)
    del bin_networks

    print('\nArtery-Vein vessel segmentation.')
    av_networks = utils.get_av_models()
    segment.arteryvein_vessel_segmentation(av_networks,
                                           AUTOMORPH_RESULTS)
    del av_networks

    print('\nOptic Disc vessel segmentation.')
    od_networks = utils.get_od_models()
    segment.opticdisc_segmentation(od_networks,
                                   AUTOMORPH_RESULTS)
    del od_networks
    print('Segmentation module complete!')

    print('\nFeature measurement module.')
    measure.feature_measurement(image_list, 
                                AUTOMORPH_RESULTS)
    print('\nAutoMorphalyzer analysis complete!')


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

    # run analysis
    run(params)
