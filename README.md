# AutoMorphalyzer 2025 👀

This codebase is a significantly adapted version from the original codebase found [here](https://rmaphoh.github.io/projects/automorph.html). The original codebase was written as part of the publication [AutoMorph: Automated Retinal Vascular Morphology Quantification via a Deep Learning Pipeline](https://tvst.arvojournals.org/article.aspx?articleid=2783477).

This adaptation was motivated by the need for a faster, more accessible, user-friendly and correct version of AutoMorph at the University of Edinburgh. We hope this may also prove useful to other research institutes.

Please contact **Jamie.Burke@ed.ac.uk** if you have questions.

**Note**: This is still in development, and further tests are required to compare the output from AutoMorph and AutoMorphalyzer.

---

## Major changes from [Automorph](https://github.com/rmaphoh/AutoMorph) in AutoMorphalyzer

* Code is restructured to run from a simple python script `automorph/main.py` directly on Anaconda Prompt after local installation.

* The end-user is only required to specify file paths in `config.txt`, requiring two inputs: the path to the data (`input_directory`) and where to save the results (`output_directory`). Upon running `python automorph/main.py' while inside the AutoMorphalyzer folder using Anaconda Prompt will trigger the pipeline to run.

* Model weights are packaged as a GitHub release and automatically downloaded upon first running locally. This will not need to be done again.

* CFP images are currently not rejected based on quality (the original M1 module has been removed), and instead are always fed through the pipeline. Try-exceptions are in place to ensure smooth batch processing.

* Quality is now computed as a probability of rejection from [QuickQual](https://github.com/justinengelmann/QuickQual) and saved out in the final results file collating all features.

* At the pre-processing stage, images are downsized to (912,912) as that is the largest dimension used for segmentation and feature measurement. This also helps facilitate manual annotation of segmentations using ITK-Snap (see end of list).

* For all modules (pre-processing, segmentation, feature measurement), if an image has previously been analysed it is skipped to save time. For each module this is detected by
  - Pre-processing: the presence of the image file path in `{output_directory}\M0\crop_csv.csv`.
  - segmentation: the presence of the segmentation mask in each of the `{output_directory}\M2\{model_type}\raw_binary` for model_type in `binary_vessel`, `artery_vein` and `optic_disc`.
  - feature measurement: the presence of the image file path in the `feature_measurements.csv` results file, assuming this already exists in `{output_directory}\M3`.

* By default, segmentation is still performed using the original models from the AutoMorph codebase. That is, 
  - Vessel segmentation (Binary vessel & Artery-Vein): [BF-Net](https://github.com/rmaphoh/Learning-AVSegmentation.git)
  - Optic disc segmentation: [lwnet](https://github.com/agaldran/lwnet.git)

* The feature measurement module has had a major revamp to remove redunancy and improve processing time. Comparisons and execution times to be published soon on this repository.
  - Fractal dimension, vessel density and global vessel calibre remain the same. These are only provided across the whole image, and **not** across zones B and C.
  - Tortuosity (distance and density) and local calibre are corrected from the original codebase. These are provided across all zones (whole image, B and C).
    - Extracting individual vessel segments has been sped up significantly through use of Numba and a depth-first search algorithm.
    - Tortuosity is calculated across individual vessel segments, and the original codebase extracted some vessels segments incorrectly, leading to exaggerated values in tortuosity. This has been corrected.
    - Local calibre was originally measured inefficiently per vessel segment.  
  - Tortuosity squared curvature has been removed due to redundancy with other tortuosity measures. 
  - We only measure CRAE and CRVE using the Knudtson formula, and remove the Hubbard formula. These are only measured in zones B and C.
  - We include arteriovenous ratio (AVR) in zones B and C.
  
*  If an error is encountered during pre-processing or feature measurement, the file is skipped and a full traceback is both printed out in the terminal during processing, and is also written to a .txt log file which is saved in `{output_directory}\M3`. If failure occurs at feature measurement, relevant metrics will be saved as -1s in the output .csv file.

* Extensive use of the `os` package throughout the codebase ensures that this pipeline will work across different OS systems. It has currently been tested on Windows and MacOS (so is likely to also work in Linux).

* Composite segmentation visualisations of the binary vessels, arteries, veins and optic disc/cup are saved out per file for easy look-up and quality inspection. This can be found in `{output_directory}\M3\segmentations`.

* AutoMorphalyzer infers laterality based on vessel density either side of the lateral position of the optic disc/cup.

* The only images saved out currently are composite segmentations (as above) and raw binary segmentations of the binary vessels, artery-veins and optic disc/cup. These are saved at the same dimension as for feature measurement `(912,912)`. Thus, we do not save out uncertainty maps (from the ensemble models), or segmentation masks for different zones or at different dimensions.

* Currently we assume no knowledge of pixel resolution, so metrics global/local vessel calibre, disc/cup height and width are left in pixel units. Both the original and new codebases measure metrics using a universal dimension of `(912,912)` so these features are still comparable across populations analysed using these codebases.

* This pipeline allows manual editing of binary vessel, artery-vein and optic disc-cup segmentations using ITK-Snap and fed back into pipeline to recompute feature measuremements. Please see the `manual_annotations/` folder for instructions on how to download and use ITK-Snap to load in segmentations for manual correction.

* This pipeline has an addition set of segmentation models besides AutoMorph's original segmentation models. To use these models instead, flag `AutoMorph_models` as `0` in `config.txt` before running the pipleime. 
  - These are segmentation models developed by Fhima, et al. for their automatic pipeline [LUNet](https://github.com/aim-lab/LUNet) and software package [PVBM](https://github.com/aim-lab/PVBM). The publications associated with this model and package are below: 
    - [LUNet: deep learning for the segmentation of arterioles and venules in high resolution fundus images](https://pubmed.ncbi.nlm.nih.gov/38599224/)
    - [PVBM: A Python Vasculature Biomarker Toolbox Based on Retinal Blood Vessel Segmentation](https://link.springer.com/chapter/10.1007/978-3-031-25066-8_15).
  - Note that these models do not perform optic cup segmentation, so any metrics relevant to the optic cup (`cup_width`, `CDR_vertical`, etc.) are saved out as -1s.

---

## Getting started

To get a local copy up, follow the steps in `quick_start.txt`, or follow the instructions below.

1. You will need a local installation of Python to run AutoMorphalyzer. We recommend a lightweight package management system such as Miniconda. Follow the instructions [here](https://docs.anaconda.com/free/miniconda/miniconda-install/) to download Miniconda for your desired operating system.

2. After downloading, navigate and open the Anaconda Prompt and clone the AutoMorphalyzer repository.

```
git clone https://github.com/jaburke166/AutoMorphalyzer.git
```

Alternatively, download the repository as a `.zip` file and extract it into a folder called `AutoMorphalyzer`.

3. Create environment and install dependencies to create your own environment in Miniconda.

```
conda create -n automorph-env python=3.11 -y
conda activate automorph-env
pip install -r requirements.txt
```

Done! You have successfully set up the software to analyse colour fundus photography data!

See below for how to run AutoMorphalyzer on your own data.

---

## Running AutoMorphalyzer


**note**: While we endeavoured to speed up the codebase, we still recommend the use of GPU acceleration for the segmentation module.

---

## Contributors

The contributors to this adapted codebase are:

* Jamie Burke (Jamie.Burke@ed.ac.uk)

The original author of the AutoMorph codebase is:

* Yukun Zhou (yukun.zhou.19@ucl.ac.uk)

The original author of the PVBM codebase is:

* Jonathan Fhima (jonathanfh@campus.technion.ac.il)

## Citing

If you wish to use this toolkit please consider citing the original work using the following BibText, with additional comment on this being an adapted version.

```
@article{zhou2022automorph,
  title={AutoMorph: Automated Retinal Vascular Morphology Quantification Via a Deep Learning Pipeline},
  author={Zhou, Yukun and Wagner, Siegfried K and Chia, Mark A and Zhao, An and Xu, Moucheng and Struyven, Robbert and Alexander, Daniel C and Keane, Pearse A and others},
  journal={Translational vision science \& technology},
  volume={11},
  number={7},
  pages={12--12},
  year={2022},
  publisher={The Association for Research in Vision and Ophthalmology}
}
```

Additionally, if you choose to use the PVBM model weights for segmentation (flagging `AutoMorph_models` as `0` in `config.txt`) then please consider citing the original work using the following BibText, with additional comment on this particular pipeline (AutoMorphalyzer) being used.

```
@inproceedings{fhima2022pvbm,
  title={PVBM: a Python vasculature biomarker toolbox based on retinal blood vessel segmentation},
  author={Fhima, Jonathan and Eijgen, Jan Van and Stalmans, Ingeborg and Men, Yevgeniy and Freiman, Moti and Behar, Joachim A},
  booktitle={European Conference on Computer Vision},
  pages={296--312},
  year={2022},
  organization={Springer}
}

@article{fhima2024lunet,
  title={LUNet: deep learning for the segmentation of arterioles and venules in high resolution fundus images},
  author={Fhima, Jonathan and Van Eijgen, Jan and Moulin-Roms{\'e}e, Marie-Isaline Billen and Brackenier, Helo{\"\i}se and Kulenovic, Hana and Debeuf, Val{\'e}rie and Vangilbergen, Marie and Freiman, Moti and Stalmans, Ingeborg and Behar, Joachim A},
  journal={Physiological Measurement},
  volume={45},
  number={5},
  pages={055002},
  year={2024},
  publisher={IOP Publishing}
}
```
