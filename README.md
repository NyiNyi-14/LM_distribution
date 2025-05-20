<!-- # LM_distribution

This repository contains all the necessary files for running the adaptive PID control simulations and related tasks.

---

## 📁 Project Structure

All source code files are organized under a single directory for ease of use.

i_PID/
├── run_main.py               # Main entry point of the project
├── requirements.txt          # List of required Python packages
├── README.md                 # This file
├── [*.py files]              # Supporting modules/scripts
└── [your output folders]     # Directories to save frames/results -->

# LM Droplet Detection and Motion Prediction via Image Processing and EDMD

This repository contains the code, data, and methods used to analyze the behavior of liquid metal (LM) droplets under substrate vibration. It includes an image processing pipeline for detecting and tracking LM droplets from microscope video sequences, and implements Extended Dynamic Mode Decomposition (EDMD) to explore their motion dynamics.

---

## 📁 Project Structure

- `videos/` – Original video files of LM droplet extrusion under vibration  
- `frames/` – Extracted and preprocessed frames  
- `DCT_IDCT.py` – Blockwise DCT and IDCT functions for denoising  
- `DnCNN.py` – Deep learning-based denoising network  
- `LM_dist.py` – Main script for image preprocessing and feature extraction  
- `LM_EDMD.py` – Script for performing EDMD analysis  
- `README.md` – Project documentation  
- `requirements.txt` – Required Python packages  

---

## 🧪 Requirements

- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- OpenCV
- scikit-image
- PyTorch (for DnCNN)

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Step 1: Setup

Download or clone the repository:

```bash
git clone https://github.com/yourusername/LM-droplet-analysis.git
cd LM-droplet-analysis
```

Make sure all scripts and videos are in the same directory or update paths accordingly in the code.

### Step 2: Image Preprocessing

Run the main image processing pipeline:

```bash
python LM_dist.py
```

- Adjust paths in `LM_dist.py` to point to your video and frame folders.
- This script will extract frames, apply DCT/IDCT, DnCNN denoising, CLAHE contrast enhancement, edge detection, and contour analysis.

### Step 3: EDMD Analysis

To perform motion prediction using EDMD:

```bash
python LM_EDMD.py
```

This script builds snapshot matrices and applies the EDMD algorithm to model and analyze droplet dynamics.

---

## 📊 Outputs

- Processed images with labeled contours and centroids
- Trajectory plots of tracked droplets
- Area and position data for each droplet
- EDMD results showing dynamic behavior

---

## 📌 Notes

- All area and position measurements are in pixel units.
- For physical unit conversion (e.g., mm²), a real-world calibration must be performed.
- Only one or two droplets were physically measured in this proof-of-concept.

---

## 📚 Related Work

This project builds on image processing and scientific computing methods, including:

- DCT-based denoising
- Deep CNN-based denoising (DnCNN)
- Edge detection & morphological operations
- Feature tracking
- Extended Dynamic Mode Decomposition (EDMD)

---

## 📥 Citation

If you use this work, please cite the related paper or include this project in your acknowledgements.

---

## 🧑‍💻 Author

**Nyi Nyi Aung** – PhD Student  
Department of Electrical and Computer Engineering  
[University Name]
