# Face Image Enhancement & Recognition Pipeline

Digital Image Processing project investigating how classical noise removal and detail sharpening methods affect image quality and face recognition accuracy under challenging lighting conditions.

## Dataset
- **Source:** Extended Yale Face Database B
- **Subset:** 5 subjects, 25 aligned grayscale face images per subject (125 images total, `256x256`)
- **Train / Test Split:** Images are sorted by lighting angle ($\sqrt{\text{Azimuth}^2 + \text{Elevation}^2}$). The 60% most frontal/well-lit images form the **training set** (75 images), and the 40% more shadowed/angled images form the **test set** (50 images).

## Project Structure
- `alignment.py` — **Module 1:** Face detection (Haar Cascade), 68-landmark eye alignment (LBF), and `256x256` cropping.
- `noise_removal.py` — **Module 2:** Synthetic Gaussian and Salt & Pepper noise injection, evaluated with Gaussian, Median, Bilateral, and Non-Local Means filters.
- `sharpening.py` — **Module 3:** Detail enhancement using Unsharp Masking and Laplacian sharpening, measured with Variance of Laplacian and Tenengrad metrics.
- `evaluation.py` — **Module 4 (Backend):** End-to-end pipeline benchmark combining quality metrics (PSNR, SSIM, Laplacian Variance, Tenengrad, Contrast) and LBPH face recognition accuracy.
- `app.py` — **Module 4 (Dashboard):** Interactive Streamlit interface for stage-by-stage visual comparison, pixel intensity histograms, full benchmarks, and failure-case inspection.
- `notebooks/main.ipynb` — Executed technical report containing all metrics, tables, plots, and error analysis.

## Installation
Install the required packages (remove `opencv-python` first if installed to avoid conflicts with `opencv-contrib-python`):

    pip install -r requirements.txt

## Usage
Launch the interactive Streamlit panel from the project root:

    python -m streamlit run app.py

Or inspect the pre-executed analysis and visualizations directly in `notebooks/main.ipynb`.

## Key Findings
- **Gaussian Noise ($\sigma = 15$):** `Bilateral + Unsharp Masking` restores facial edges effectively while preserving structure and maintaining high LBPH recognition accuracy.
- **Salt & Pepper Noise ($2\%$):** `Median Filter` clearly outperforms other filters (**43.03 dB PSNR**, **0.9923 SSIM**, **90.0% LBPH accuracy**), whereas Bilateral and NL-Means treat impulse dots as edges.
- **Error Analysis:** Misclassifications concentrate in extreme side-lighting angles (`70°–90°`) and in combinations where aggressive sharpening (e.g., Laplacian on residual noise) amplifies noise instead of facial features.