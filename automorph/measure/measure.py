import os
import sys

SCRIPT_PATH = os.path.realpath(os.path.dirname(__file__))
MODULE_PATH = os.path.split(SCRIPT_PATH)[0]
PACKAGE_PATH = os.path.split(MODULE_PATH)[0]
sys.path.append(SCRIPT_PATH)
sys.path.append(MODULE_PATH)
sys.path.append(PACKAGE_PATH)

import numpy as np
import cv2
import bottleneck as bn
import utils
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from skimage.draw import polygon
from skimage import morphology, measure, transform
from tqdm import tqdm

import get_vessel_coords
from preprocess import fundus_prep as prep

DISC_COLS = ['laterality', 'macular_centred', 'disc_height', 'disc_width', 'cup_height', 'cup_width', 'CDR_vertical', 'CDR_horizontal']

VESSEL_COLS = ['vessel_density', 'fractal_dimension', 'average_global_calibre', 'average_local_calibre', 'tortuosity_distance', 'tortuosity_density'] 
LARGE_VESSEL_COLS = ['CRAE_Knudtson_artery_B', 'CRAE_Knudtson_artery_C', 'CRVE_Knudtson_vein_B', 'CRVE_Knudtson_vein_C', 'AVR_B', 'AVR_C']
ALL_COLS = [f'{f}_{v}_{z}'for z in ['whole', 'B', 'C']  for v in ['binary', 'artery', 'vein'] for f in VESSEL_COLS] + LARGE_VESSEL_COLS


def boxcount(Z, k):
    S = np.add.reduceat(np.add.reduceat(Z,
                        np.arange(0, Z.shape[0], k), axis=0),
                        np.arange(0, Z.shape[1], k), axis=1)
    return len(np.where((S > 0) & (S < k*k))[0])



def fractal_dimension(Z):

    p = min(Z.shape)
    n = 2**np.floor(np.log(p)/np.log(2))
    n = int(np.log(n)/np.log(2))
    sizes = 2**np.arange(n, 1, -1)
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))

    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]



def global_metrics(vessel_, skeleton):

    vessel_ = (vessel_ > 0).astype(int)
    skeleton = (skeleton > 0).astype(int)
        
    fractal_d = fractal_dimension(vessel_)
    global_width = np.sum(vessel_)/np.sum(skeleton)
    
    return fractal_d, global_width



def Hubbard_cal(w1,w2):
    w_artery = np.sqrt(0.87*np.square(w1) + 1.01*np.square(w2) - 0.22*w1*w2 - 10.76) 
    w_vein = np.sqrt(0.72*np.square(w1)+0.91*np.square(w2)+450.05)
    return w_artery, w_vein



def Knudtson_cal(w1,w2):
    w_artery = 0.88*np.sqrt(np.square(w1) + np.square(w2)) 
    w_vein = 0.95*np.sqrt(np.square(w1) + np.square(w2)) 
    return w_artery, w_vein



def _distance_2p(x1, y1, x2, y2):
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5



def _curve_length(x, y):
    return np.sum(((x[:-1] - x[1:]) ** 2 + (y[:-1] - y[1:]) ** 2) ** 0.5)



def _chord_length(x, y):
    return _distance_2p(x[0], y[0], x[len(x) - 1], y[len(y) - 1])



def _detect_inflection_points(x, y):
    cf = np.convolve(y, [1, -1])
    inflection_points = []
    for i in range(2, len(x)):
        if np.sign(cf[i]) != np.sign(cf[i - 1]):
            inflection_points.append(i - 1)
    return inflection_points



def tortuosity_density(x, y, curve_length):
    inflection_points = _detect_inflection_points(x, y)
    n = len(inflection_points)
    if not n:
        return 0
    starting_position = 0
    sum_segments = 0
    
    # we process the curve dividing it on its inflection points
    for in_point in inflection_points:
        segment_x = x[starting_position:in_point]
        segment_y = y[starting_position:in_point]
        seg_curve = _curve_length(segment_x, segment_y)
        seg_chord = _chord_length(segment_x, segment_y)
        if seg_chord:
            sum_segments += seg_curve / seg_chord - 1
        starting_position = in_point

    # return ((n - 1)/curve_length)*sum_segments  # This is the proper formula
    return (n - 1)/n + (1/curve_length)*sum_segments # This is not



def _refine_coords(coords: list[np.ndarray], dtype: type = np.int16):
    return [_refine_path(c).astype(dtype) for c in coords]



def _refine_path(data: np.ndarray, window: int = 4):
    # Simple moving average
    return bn.move_mean(data, window=window, axis=0, min_count=1)


    
def _compute_vessel_edges(coords: list[np.ndarray], dist_map: np.ndarray):
    edges1 = []
    edges2 = []
    for path in coords:
        x, y = path[:,0], path[:,1]
        delta = np.gradient(path, axis=0)
        angles = np.arctan2(delta[:,1], delta[:,0])
        d = dist_map[x, y]
        offset_x = d * np.cos(angles + np.pi/2)
        offset_y = d * np.sin(angles + np.pi/2)
        x_edge1 = x + offset_x
        y_edge1 = y + offset_y
        x_edge2 = x - offset_x
        y_edge2 = y - offset_y
        edges1.append(np.stack([x_edge1, y_edge1], axis=1))
        edges2.append(np.stack([x_edge2, y_edge2], axis=1))
        
    return edges1, edges2


    
def _calculate_vessel_widths(mask, coords):
    
    # Refine coordinates
    coords_refined = _refine_coords(coords) # dtype = np.int16
    
    # Distance transform
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    
    # Get diameter of refined vessel skeleton
    vessel_map = np.zeros_like(dist, dtype=bool)
    c_r = np.concatenate(coords_refined, axis=0)
    vessel_map[c_r[:,0], c_r[:,1]] = True
    vessel_map = dist * vessel_map + 2.0
    
    # Calculate edges of the vessels
    edges1, edges2 = _compute_vessel_edges(coords_refined, vessel_map)
    
    # Binary image of pixels within the edges
    mask_edges = np.zeros_like(mask, dtype=bool)
    for edge1, edge2 in zip(edges1, edges2):
        combined = np.vstack((edge1, edge2[::-1]))
        rr, cc = polygon(combined[:, 0], combined[:, 1], shape=mask.shape)
        mask_edges[rr, cc] = True
        
    # AND with segmentation mask
    mask_edges = mask_edges & (mask > 0)
    
    # Identify the edges of the vessels (Canny edge detection)
    mask_edges = cv2.Canny(mask_edges.astype(np.uint8), 0, 1)
    
    # Locate edges in the original image for each vessel
    on_pixels = np.argwhere(mask_edges).astype(np.float32)
    tree = KDTree(on_pixels)
    edges1 = [on_pixels[tree.query(e)[1]] for e in edges1]
    edges2 = [on_pixels[tree.query(e)[1]] for e in edges2]
    
    # Calculate vessel width at each point + average width
    widths = [np.linalg.norm(e1 - e2, axis=1) for e1, e2 in zip(edges1, edges2)]
    avg_width = [np.mean(w, dtype=float) for w in widths]
    
    return avg_width




def vessel_metrics(vessels,
                   vessel_coords,
                   roi_masks,
                   min_pixels_per_vessel: int = 10, 
                   vessel_type: str = "binary"):
    """
    Re-write of tortuosity_measures.evaluate_window() to include only necessary code.
    """
    # collect outputs in a dictionary
    vessels = (vessels > 0).astype(np.uint8)
    cfp_dict = {'whole':{}, 'B':{}, 'C':{}}
    logging_list = []

    # Number of vessel pixels
    vessel_total_count = np.sum(vessels==1) 
    pixel_total_count = vessels.shape[0]*vessels.shape[1]
    vessel_density = vessel_total_count / pixel_total_count

    # Compute FD, VD and Average width over whole image
    skeleton = morphology.skeletonize(vessels)
    fractal_dimension, average_width_all = global_metrics(vessels, skeleton) 
    cfp_dict['whole']["fractal_dimension"] = fractal_dimension
    cfp_dict['whole']["vessel_density"] = vessel_density
    cfp_dict['whole']["average_global_calibre"] = average_width_all
    
    for roi_type, mask in roi_masks.items():

        # Get zonal vessel map
        zonal_mask = (vessels * mask).astype(np.uint8)
        
        # Loop over windows
        tcurve = 0
        tcc = 0
        td = 0
        
        # Initialise vessel widths and count lists
        vessel_count = 0
        zonal_vessels = []
        for i, vessel in enumerate(vessel_coords):

            if roi_type in ['B', 'C']:
                idx_in_zone = np.where(zonal_mask[vessel[:,0], vessel[:,1]])[0]
                if idx_in_zone.shape[0] < min_pixels_per_vessel:
                    continue
                else:
                    vessel = vessel[idx_in_zone]
            vessel_count += 1
            zonal_vessels.append(vessel)
                         
            # Work out length of current vessel
            vessel = vessel.T
            N = len(vessel[0])
            v_length = _curve_length(vessel[0], vessel[1])
            c_length = _chord_length(vessel[0], vessel[1])
            tcc += v_length / c_length
                    
            # tcurve is simply the pixel length of the vessel
            tcurve += v_length
            
            # td measures curve_chord_ratio for subvessel segments per inflection point 
            # and cumulatively add them, and scale by number of inflections and overall curve length
            # formula comes from https://ieeexplore.ieee.org/document/1279902
            td += tortuosity_density(vessel[0], vessel[1], v_length)

        # Normalise tortuosity density and tortuosity distance by vessel_count
        td = td/vessel_count
        tcc = tcc/vessel_count
    
        # This is measuring the same thing as average_width computed in global_metrics, but should be smaller as 
        # individual vessel segments exclude branching points in their calculation
        all_vessel_widths = _calculate_vessel_widths(zonal_mask, zonal_vessels)
        local_caliber = np.mean(all_vessel_widths)
        
        # collect outputs
        cfp_dict[roi_type]["tortuosity_density"] = td
        cfp_dict[roi_type]['tortuosity_distance'] = tcc
        cfp_dict[roi_type]["average_local_calibre"] = local_caliber

        # Do not calculate CRAE/CRVE if binary vessels.
        #print('CRAE/CRVE')
        if (vessel_type != "binary") & (roi_type in ['B', 'C']):   
        
            # calculate the CRAE/CRVE with Knudtson calibre
            vtype = vessel_type[0].upper()
            sorted_vessel_widths_average = sorted(all_vessel_widths)[-6:]
            N_vessels = len(sorted_vessel_widths_average)
        
            # Error handle if detected less than 6 vessels, must be even number
            if N_vessels < 6:
                msg1 = f'        WARNING: Less than 6 vessels detected in zone. Please check segmentation. Returning -1 for CR{vtype}E.'
                msg2 = f'                 Note that this means AVR cannot be computed for this image'
                cfp_dict[roi_type]["CRAE_Knudtson"] = -1
                cfp_dict[roi_type]["CRVE_Knudtson"] = -1
        
                # log to user
                print(msg1)
                print(msg2)
                logging_list.append(msg1)
                logging_list.append(msg2)
        
            #  Compute calibre, taking into account number of available vessels
            else:
                
                w_first_artery_Knudtson_1, w_first_vein_Knudtson_1 = Knudtson_cal(sorted_vessel_widths_average[0],
                                                                                  sorted_vessel_widths_average[5])
                
                w_first_artery_Knudtson_2, w_first_vein_Knudtson_2 = Knudtson_cal(sorted_vessel_widths_average[1],
                                                                                  sorted_vessel_widths_average[4])
                    
                w_first_artery_Knudtson_3, w_first_vein_Knudtson_3 = Knudtson_cal(sorted_vessel_widths_average[2],
                                                                                  sorted_vessel_widths_average[3])
                
                CRAE_first_round = sorted([w_first_artery_Knudtson_1,
                                           w_first_artery_Knudtson_2,
                                           w_first_artery_Knudtson_3])
                CRVE_first_round = sorted([w_first_vein_Knudtson_1,
                                           w_first_vein_Knudtson_2,
                                           w_first_vein_Knudtson_3])
                
                if vessel_type=='artery': 
                    w_second_artery_Knudtson_1, w_second_vein_Knudtson_1 = Knudtson_cal(CRAE_first_round[0],
                                                                                        CRAE_first_round[2])  
                
                    CRAE_second_round = sorted([w_second_artery_Knudtson_1,CRAE_first_round[1]])
                    CRAE_Knudtson,_ = Knudtson_cal(CRAE_second_round[0],CRAE_second_round[1])
                    cfp_dict[roi_type]["CRAE_Knudtson"] = CRAE_Knudtson
                    cfp_dict[roi_type]["CRVE_Knudtson"] = -1
                
                else:
                    w_second_artery_Knudtson_1, w_second_vein_Knudtson_1 = Knudtson_cal(CRVE_first_round[0],
                                                                                        CRVE_first_round[2])  
                
                    CRVE_second_round = sorted([w_second_vein_Knudtson_1,CRVE_first_round[1]])
                    _,CRVE_Knudtson = Knudtson_cal(CRVE_second_round[0],CRVE_second_round[1])
                    cfp_dict[roi_type]["CRAE_Knudtson"] = -1
                    cfp_dict[roi_type]["CRVE_Knudtson"] = CRVE_Knudtson
                
        else:
            cfp_dict[roi_type]["CRAE_Knudtson"] = -1
            cfp_dict[roi_type]["CRVE_Knudtson"] = -1
        
        
    return cfp_dict



def get_disc_metrics(disc_mask, cup_mask, vessel_mask):

    # Empty dictionary of metrics
    disc_metrics = {}

    # Metrics for the optic disc
    img_size = disc_mask.shape[0]
    try:
        disc_index = np.where(disc_mask>0)
        disc_index_x = disc_index[1]
        disc_index_y = disc_index[0]
        disc_metrics['disc_height'] = np.max(disc_index_y)-np.min(disc_index_y)
        disc_metrics['disc_width'] = np.max(disc_index_x)-np.min(disc_index_x)
    except:
        disc_metrics['cup_height'] = -1
        disc_metrics['cup_width'] = -1

    # Metrics for the optic cup
    try:
        cup_index = np.where(cup_mask>0)
        cup_index_x = cup_index[1]
        cup_index_y = cup_index[0]
        disc_metrics['cup_height'] = np.max(cup_index_y)-np.min(cup_index_y)
        disc_metrics['cup_width'] = np.max(cup_index_x)-np.min(cup_index_x)
    except:
        disc_metrics['cup_height'] = -1
        disc_metrics['cup_width'] = -1

    # Measure cup-to-disc ratios
    try:
        disc_metrics['CDR_vertical'] = disc_metrics['cup_height']/disc_metrics['disc_height']
        disc_metrics['CDR_horizontal'] = disc_metrics['cup_width']/disc_metrics['disc_width']
    except:
        disc_metrics['CDR_vertical'] = -1
        disc_metrics['CDR_horizontal'] = -1

    # Centre of the optic cup (and thus disc)
    try:
        cup_centre = measure.centroid(cup_mask)
        disc_metrics['cup_centre_x'] = int(cup_centre[1])
        disc_metrics['cup_centre_y'] = int(cup_centre[0])
    
        # Sense checking the detected disc and cup metrics
        condition = disc_metrics['disc_width']<(img_size/3) and \
                    disc_metrics['disc_height']<(img_size/3) and \
                    disc_metrics['cup_centre_x']<=np.max(disc_index_x) and \
                    disc_metrics['cup_centre_x']>=np.min(disc_index_x) and \
                    disc_metrics['cup_centre_y']<=np.max(disc_index_y) and \
                    disc_metrics['cup_centre_y']>=np.min(disc_index_y) and \
                    disc_metrics['cup_height']<disc_metrics['disc_height'] and \
                    disc_metrics['cup_width']<disc_metrics['disc_width']
    
        # Work out centering of the scan, i.e. whether the optic disc is centred in the image
        # We define this as the optic disc being within 10% of the image centre
        disc_metrics['macular_centred'] = True
        if condition:
            horizontal_distance = np.absolute(np.mean(disc_index_y)-img_size/2)
            vertical_distance = np.absolute(np.mean(disc_index_x)-img_size/2)
            distance_ = np.sqrt(np.square(horizontal_distance)+np.square(vertical_distance))
            if (distance_/img_size)<0.1:
                disc_metrics['macular_centred'] = False
    
        # Infer laterality based on how much vessel was detected either side of the disc centre, i.e. 
        # regardless of centring, there should be more vasculature toward the macula than the temporal side
        disc_metrics['laterality'] = 'Left'
        if vessel_mask[:, :disc_metrics['cup_centre_x']].sum() > vessel_mask[:, disc_metrics['cup_centre_x']:].sum():
            disc_metrics['laterality'] = 'Right'
    except:
        disc_metrics['macular_centred'] = -1
        disc_metrics['laterality'] = -1
        disc_metrics['cup_centre_x'] = -1
        disc_metrics['cup_centre_y'] = -1
        

    return disc_metrics



def merge_results(all_cfp_dict, all_disc_metrics, output_directory):
    
    # Process dataframe for accessibility and readability
    feature_df = utils.nested_dict_to_df(all_cfp_dict).reset_index()
    feature_df.columns = ['Filename', 'vessel_type', 'zone'] + list(feature_df.columns[3:])
    feature_df = feature_df.pivot_table(index=['Filename'], columns=['vessel_type','zone'], values=feature_df.columns[3:]).reset_index()
    feature_df.columns = ['_'.join(col).strip() for col in feature_df.columns.values]
    feature_df = feature_df[feature_df.columns[~np.all(feature_df == -1, axis=0).values]]

    # Add in Arteriovenule ratio with Try-Except blocks
    try:
        feature_df['AVR_B'] = feature_df.CRAE_Knudtson_artery_B / feature_df.CRVE_Knudtson_vein_B
    except:
        feature_df['AVR_B'] = -1
    try:
        feature_df['AVR_C'] = feature_df.CRAE_Knudtson_artery_C / feature_df.CRVE_Knudtson_vein_C
    except:
        feature_df['AVR_C'] = -1
    
    # Set AVRs to 1 where any one of CRAE/CRVE failed
    feature_df['AVR_B'] = [-1 if avr < 0 else (-1 if avr==-1 else avr) for avr in feature_df['AVR_B']]
    feature_df['AVR_C'] = [-1 if avr < 0 else (-1 if avr==-1 else avr) for avr in feature_df['AVR_C']]

    # Order columns via ALL_COLS (see start of script)
    col_order = [col for col in ALL_COLS if col in feature_df.columns]
    feature_df = feature_df[['Filename__'] + col_order]
    feature_df = feature_df.rename({'Filename__':'Filename'}, axis=1)

    # Organise disc metrics (see start of script for DISC_COLS)
    disc_df = utils.nested_dict_to_df(all_disc_metrics).reset_index()
    disc_df.rename({'index':'Filename'}, axis=1, inplace=True)
    disc_df = disc_df[['Filename']+DISC_COLS]

    # Merge with quality
    quickqual_df = pd.read_csv(os.path.join(output_directory, 'M0', 'crop_info.csv'))[['Name', 'quality']]
    quickqual_df['Name'] = quickqual_df['Name'].str.split('.',expand=True)[0] + '.' + disc_df.Filename.str.split('.',expand=True)[1]
    quickqual_df.rename({'Name':'Filename', 'quality':'QuickQual_quality'}, axis=1, inplace=True)

    # Merge dataframes
    metric_df = (disc_df.merge(quickqual_df, on='Filename')).merge(feature_df, on='Filename')

    return metric_df



def df_to_dict(img_feat_df):

    vessel_types = ['binary', 'artery', 'vein']
    zone_types = ['whole', 'B', 'C']
    img_all_dict = {vtype:{zone:{} for zone in zone_types} for vtype in vessel_types}
    for col in img_feat_df.columns:
        if 'AVR' in col:
            continue
        vessel_col = '_'.join(col.split('_')[:2]) if 'calibre' not in col else '_'.join(col.split('_')[:3])
        value = np.unique(img_feat_df[col])[0]
        for vtype in vessel_types:
            if f'_{vtype}' in col:
                for zone in zone_types:
                    if f'_{zone}' in col:
                        img_all_dict[vtype][zone][vessel_col] = value

    return img_all_dict



def feature_measurement(image_list, output_directory):

    # Directory structure
    save_path = os.path.join(output_directory, 'M3', 'segmentations')
    image_directory = os.path.join(output_directory, 'M0', 'images')
    segmentation_directory = os.path.join(output_directory, 'M2')
    annotate_directory = os.path.join(output_directory, 'annotations')

    # Check for manual annotations
    annot_list = set(Path(annotate_directory).glob('*.nii.gz'))
    used_list = set(Path(annotate_directory).glob('*used.nii.gz'))
    annot_list = list(annot_list - used_list)

    # Create directory to store superimposed segmentations
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Check if feature_measurements exist already and load in if so
    results_path = os.path.join(output_directory, 'M3', 'feature_measurements.csv')
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
    else:
        results_df = None

    # Loop over images
    all_disc_metrics = {} 
    all_cfp_dict = {}
    img_size = (912,912)
    N = len(image_list)
    problem_log = []
    for i, img_name in tqdm(enumerate(image_list), total=N, desc='Visualisation and feature measurement', unit='image'):

        try:
            # Extract filename
            fname = os.path.splitext(img_name)[0]
                    
            # Load in image (already at (912,912) dimensions)
            cfp_img = prep.imread(os.path.join(image_directory, fname + '.png'))

            # Check for any manual annotations and load in, otherwise load in original segmentation masks
            binmask_path = os.path.join(annotate_directory, f'{fname}_binary_vessel.nii.gz')
            av_path = os.path.join(annotate_directory, f'{fname}_artery_vein.nii.gz')
            od_path = os.path.join(annotate_directory, f'{fname}_optic_disc.nii.gz')
            segmentation_masks = []
            v_manuals = []
            manual_annot = False
            for path, vtype in zip([binmask_path, av_path, od_path], ['binary_vessel', 'artery_vein', 'optic_disc']):
                if os.path.exists(path):
                    v_manuals.append(vtype)
                    manual_annot = True
                    mask = utils.load_annotation(path, vtype).astype(bool)
                    if os.path.exists(path.split(".")[0]+"_used.nii.gz"):
                        os.remove(path.split(".")[0]+"_used.nii.gz")
                    os.rename(path, path.split(".")[0]+"_used.nii.gz")
                else:
                    mask = prep.imread(os.path.join(segmentation_directory, vtype, 'raw_binary', fname + '.png'))[...,0]
                    mask = utils.load_annotation(mask, vtype).astype(bool)
                segmentation_masks.append(mask)
            binmask, avmask, odmask = segmentation_masks

            # Print out to end-user of detection of manual annotation
            if manual_annot:
                v_str = ',' .join(v_manuals)
                print(f'Detected {v_str} manual annotations for {img_name}. Using these for feature measurement.')

            # Post-process disc segmentation to obtain radius and centre
            disc_mask = (255*utils.select_largest_mask(np.sum(odmask[...,:2], axis=-1))).astype(np.uint8)
            cup_mask = (255*utils.select_largest_mask(odmask[...,1])).astype(np.uint8)
            od_radius = utils.process_opticdisc(disc_mask)
            od_centre = measure.centroid(disc_mask)[[1,0]]

            # Check to see if results exist for this file and skip if so
            prev_analysed = False
            if results_df is not None and not manual_annot:
                if img_name in results_df.Filename.values:
                    prev_analysed = True
                    # print(f'{img_name} previously measured. Skipping!')
                    img_df = results_df[results_df.Filename == img_name]
                    all_disc_metrics[img_name] = img_df[DISC_COLS].iloc[0].to_dict()
                    
                    img_feat_df = img_df[[col for col in ALL_COLS if col in img_df.columns]]
                    all_cfp_dict[img_name] = df_to_dict(img_feat_df)

                    continue

            # Superimpose segmentations together and save out, only if not previously analysed or manual annotation
            if not prev_analysed or manual_annot:
                utils.superimpose_segmentations(cfp_img, binmask, avmask, disc_mask, cup_mask, save_path, fname)

            # Create dictionary to store disc metrics for each image
            all_disc_metrics[img_name] = get_disc_metrics(disc_mask, cup_mask, binmask)

            # Create dictionary to store vessel metrics for each image
            all_cfp_dict[img_name] = {}
            cfp_keys = ['binary', 'artery', 'vein']

            # Extract vessel maps
            cfp_vbinmap = binmask.copy()
            artery_vbinmap = avmask[:,:,[0,1]].sum(axis=-1) * ~disc_mask.astype(bool)
            vein_vbinmap = avmask[:,:,[2,1]].sum(axis=-1) * ~disc_mask.astype(bool)

            # Create fullest binary vessel map by adding AV-maps
            cfp_vbinmap = (cfp_vbinmap.astype(bool) + artery_vbinmap.astype(bool) + vein_vbinmap.astype(bool)) * ~disc_mask.astype(bool)

            # Generate zones to measure in
            mask_rois = utils.generate_zonal_masks(img_size, od_radius, od_centre)

            for v_map, v_type in zip([cfp_vbinmap, artery_vbinmap, vein_vbinmap], cfp_keys):

                # detect individual vessels, similar to skelentonisation but detects individual vessels, and
                # splits them at any observed intersection
                vcoords = get_vessel_coords.generate_vessel_skeleton(v_map, disc_mask, od_centre, min_length=10)
                    
                # log to user 
                all_cfp_dict[img_name][v_type] = {}
                msg = f"Measuring {v_type} vessel map"

                # Compute features 
                all_cfp_dict[img_name][v_type] = vessel_metrics(v_map,
                                                        vcoords,
                                                        mask_rois,
                                                        vessel_type=v_type)
        except Exception as e:
            msg = f"\n\nError measuring {img_name}. Outputting error recieved:"
            problem_log.append(msg)
            print(msg)
            log = utils.print_error(e, verbose=True)
            problem_log.extend(log)
            continue
            
    # Collect all metrics to save out
    metric_df = merge_results(all_cfp_dict, all_disc_metrics, output_directory)

    # Save out log storing any problems
    if len(problem_log) > 0:
        with open(os.path.join(output_directory, 'M3', 'feature_measurement_log.txt'), 'w') as f:
            for item in problem_log:
                f.write("%s\n" % item)

    # Save out dataframe
    metric_df.to_csv(os.path.join(output_directory, 'M3', 'feature_measurements.csv'), index=False)
