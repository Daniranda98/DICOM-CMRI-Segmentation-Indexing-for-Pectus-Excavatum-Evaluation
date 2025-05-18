# DICOM-CMRI-Segmentation-Indexing-for-Pectus-Excavatum-Evaluation

# Thoracic Morphometry on DICOM Images

This project performs automated analysis of chest CMRI images to calculate morphometric indices related to conditions such as Pectus Excavatum. The tool processes DICOM images to segment the thoracic cavity and compute clinically relevant parameters such as the Haller index, depression index, flattening index, and more.



## Requirements

Make sure you have **Python 3.8+** installed. You can install all required packages using:

'pip install -r requirements.txt'


## How to Use

1. **Run the main script:**

'python main.py'

2. A file dialog will prompt you to select a DICOM folder.
3. The script will:
   - Load and preprocess the images
   - Segment the thoracic cavity, lungs, and optionally the heart
   - Compute various morphometric indices
   - Save an Excel file with results in `output/results.xlsx`
   

##  Computed Indices

- **Haller Index**
- **Flattening Index**
- **Depression Index**
- **Inner Index (corrected internal area)**
- **Depression Fraction**
- **Thoracic Depression Volume**
- **Asymmetry Index**

>  Some metrics are computed only for male patients, depending on clinical criteria.

##  Extendability

The project is modular and easily extendable. You can add your own analysis modules in the `Functions/` folder.

## Notes

- Images must be axial chest CMRI scans in DICOM format.
- Proper image orientation is assumed.
- For clinical use, results should be reviewed and validated by experienced radiologists.

## License

Open-source project intended for academic and clinical research. Feel free to modify and reuse with appropriate attribution.

## Authors

- Daniele Randazzo
- Academic support by 'Università degli studi di Genova'
