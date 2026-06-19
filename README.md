# Computational_Imaging
...
## 0. Data Degradation

This notebook generates a degraded version of a subset of the ImageNet-1K dataset to be used in subsequent image restoration and inverse problem experiments.

The pipeline:

1. Loads selected ImageNet classes.
2. Applies a forward degradation model consisting of:

   * Gaussian blur
   * Additive Gaussian noise
3. Creates multiple noisy observations for each image.
4. Splits the resulting dataset into training, validation, and test sets.
5. Saves the dataset to Google Drive.
6. Performs consistency and quality checks on the generated data.

The generated dataset follows the degradation model:


$$y = K(x) + e$$


where:

* (x) = clean image
* (K) = blur operator
* (e) = additive Gaussian noise
* (y) = degraded observation

The notebook assumes that the custom IPPy package is available inside:

```text
/content/drive/MyDrive/Computational Imaging
```

### Reproducibility

The notebook enforces deterministic execution through:

```python
set_seed(seed)
```

The following generators are seeded:

* Python random
* NumPy
* PyTorch CPU
* PyTorch CUDA

Each image/noise pair receives a deterministic seed:

```python
base_seed + image_index + noise_level
```

This guarantees reproducible degradation across executions.

### Dataset Splitting

The degraded dataset is split into:

| Split      | Percentage |
| ---------- | ---------- |
| Train      | 80%        |
| Validation | 10%        |
| Test       | 10%        |

The split is stratified by class label:

```python
stratify_by_column="label"
```

to preserve class balance.

Files saved:

```text
dataset/
├── train/
├── validation/
├── test/
└── labels.json
```

where ```labels.json``` is a mapping from ImageNet class ID to human-readable class name.

Example:

```json
{
  "0": "tench",
  "10": "brambling",
  ...
}
```

This file is used later for visualization and interpretation.

### Data Verification
* Split Size Verification
* Class Distribution Verification
* Dataset Structure Verification
* Pixel Range Verification
* Noise Verification, 

---
# 1. TvP Chambolle-Pock

This repository/directory contains the Jupyter notebook `1.TvP_Chambolle_Pock.ipynb`, which implements and evaluates **Deblurring** and **Denoising** algorithms on images using the **Total p-Variation (TpV)** model.

Specifically, the notebook focuses on solving inverse image reconstruction problems by combining a data fidelity term with a non-convex Total p-Variation regularization ($p \le 1$). The optimization is performed using the **Chambolle-Pock** algorithm (Primal-Dual Hybrid Gradient - PDHG) combined with Iteratively Reweighted L1 (IR-PDHG) to handle the non-convexity introduced by $p < 1$.

## Notebook Contents

The notebook is structured into the following main sections:

1. **Libraries, Drive Access & Constants**:
   Imports the necessary libraries (including `torch`, `torchvision`, `numpy`, `matplotlib`, and `datasets`), mounts Google Drive to access the datasets, and imports custom modules (`IPPy`) for the blurring and spatial gradient operators.

2. **Basic Configuration and Data Loading**:
   Defines the main parameters for image degradation (blur kernel type and size, Gaussian noise levels). Loads the previously degraded `benjamin-paine/imagenet-1k-256x256` dataset, with verification prints for minimum/maximum values, class distributions, and error metrics (MSE, noise std). Shows verification plots of the applied degradation.

3. **TpV Function Definition (Basic Gradient Descent)**:
   Baseline implementation (`deblur_denoise_tpv`) for Total p-Variation that computes the VJP (Vector-Jacobian Product) relying on the Adam optimizer and standard Autograd for optimization.

4. **TpV Function Definition with Chambolle-Pock Algorithm**:
   The algorithmic core of the notebook (`deblur_denoise_tpv_chambolle_pock`). It implements the PDHG method fully leveraging:
   - **Dual Variables** for data fidelity and Total Variation.
   - **Reweighting Factor Computation**: Approximation of the non-convex $L_p$ term (for $p < 1$) via a weighted $L_1$ norm (Iteratively Reweighted L1).
   - **Extrapolation** (Over-relaxation) to guarantee theoretical convergence of $O(1/N)$.
   - Exact mathematical adjoint operators extraction via autograd (`backward()` for VJP).

5. **Deblurring and Denoising Test (Single Image)**:
   Test run on a single noisy image ($y_{100}$ with high noise). Produces comparative plots showing the original image (ground truth), the degraded image (with baseline PSNR/SSIM), and the reconstructed image. Also plots the learning curves (Loss, PSNR, SSIM).

6. **2D Evaluation of the "p" Parameter (Single Image)**:
   Iterative execution of the algorithm on a fixed image, uniformly varying the value of $p \in [0.1, 0.5]$ to analyze which guarantees the best results in terms of visual quality metrics (PSNR, SSIM).

7. **2D Evaluation of the Regularization Parameter "$\lambda_{tpv}$" (Single Image)**:
   Keeping $p$ fixed (e.g., $p=0.5$), the algorithm tests various values for the regularization weight $\lambda_{tpv}$ and plots the learning curves.

8. **3D Evaluation of "$\lambda_{tpv}$" (Multiple Images)**:
   Advanced testing and tuning phase. The optimization for various $\lambda_{tpv}$ values is extended over the entire test set (multiple images). 
   - Generates 3D surface plots showing the trend of PSNR and SSIM as the image index and the $\lambda$ parameter vary (on a logarithmic scale).
   - Computes and prints aggregated global statistics (Mean, Median, Mode, Standard Deviation) to robustly determine the ideal $\lambda$ value to use.

## Dependencies and Requirements

- `torch`, `torchvision` (with CUDA support recommended)
- `numpy`, `scipy`, `matplotlib`, `scikit-image`
- `datasets` (HuggingFace) for efficient arrow dataset management.
- Custom `IPPy` library (located at `/content/drive/MyDrive/Computational Imaging/IPPy`).
- `astra-toolbox` package installed via pip at the beginning of the notebook.

## Setup and Usage

The notebook is originally designed to be run on Google Colab, leveraging a mounted Google Drive containing the `Computational Imaging` directory, which is assumed to have:
- The pre-processed/saved datasets located in `.../Computational Imaging/dataset`.
- The `IPPy` module.

If you run it locally:
1. Make sure you have cloned/downloaded the `IPPy` package in your working directory (or add it to your `sys.path`).
2. Adjust the `BASE_DIR` constant to point to the correct path containing your data, or comment out the Google Drive mount (`drive.mount(...)`).
3. Make sure your environment has all the mentioned dependencies installed.

## Metrics Information

The deblur and denoise performance is evaluated throughout the optimization using:
- **PSNR** (Peak Signal-to-Noise Ratio): measured in decibels (dB), indicates the ratio between the maximum possible signal power and the power of the corrupting noise.
- **SSIM** (Structural Similarity Index): measured between $[0, 1]$, quantifies image degradation as perceived change in structural information. Values closer to 1 indicate an excellent reconstruction.
