# Retinal components detection method

An algorithm destined to identify, separate and segment the main components of the retina using fundus images. This algorithm is created using _Python_ and _OpenCv_ for the image processing.

This method is composed by three sub-methods, which focus on detecting and segmenting each of the main anatomical components found in the retina. 
The three sub-methods found in the algorithm are as follows:
1. Vascular tree detection and segmentation
>The detection of the vascular tree in the retina using a *Random Forest* model in order to do the classification between the non-vascular elements and the vascular elements. The model was trained using the [DRIVE datasets](https://drive.grand-challenge.org/) for the supervised training reference. This method is based on the work directed by *(GeethaRamani and Balasubramanian,2016)*. [^1]

2. Optic Disc detection and segmentation
>The detection of the Optic Disc structure in the retina using a voting mechanism based on statistical analysis and color intensity analysis. This method is based on the work directed by *(Aquino et al., 2010)*. [^2]

3. Macula detection and segmentation
>The detection of the Macula component based on color and morphological analysis on high contrast elements on the retina. This method is based on the work directed by *(GeethaRamani and Balasubramanian,2018)*. [^3]

Each of the sub-methods is developed in its own *.py* file, found in the **master** branch, and are combined in the class file *ret_fundus.py* in order obtain the retinal elements and segment each of them. The result is an object with a dictionary structure of each element containing its segmented mask, center location and radius for the circular elements.

# Datasets used for this project
The datasets used for this project is a recopilation of sets of accepted images from different datasets. The complete datasets are found [here](https://drive.google.com/drive/folders/18CM5gA1PsygCIoCX7RY42aGaXmZOenxW?usp=sharing), while the public datasets are found as follows:

* [DRIVE datasets](https://drive.grand-challenge.org/)

# Footnotes and further considerations
## To-do list:
- [x] Set a functional detection and segmentation method for the anatomical components in the retina.
- [x] Find a way to save the segmented components in a data structure for future references.
- [ ] Correct the algorithm to detect pathological elements in the fundus images.
- [ ] Complete the anatomical estimation search for the vascular arch and the fovea location estimation.
- [ ] Set up the database architecture for resulting structures storing and display.
- [ ] Generate a GUI application for display and control of the resulting algorithms.

## Footnotes:
[^1]: Retinal blood vessel segmentation employing image processing and data mining techniques for computerized retinal image analysis

[^2]: Detecting the optic disc boundary in digital fundus images usingmorphological, edge detection, andfeature extraction techniques

[^3]: Macula segmentation and fovealocalization employing image processing and heuristic basedclustering for automated retinalscreening
