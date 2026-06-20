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

## 1. TvP Chambolle-Pock

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

---

## 2. End-to-End Model

This notebook implements a complete image restoration pipeline based on a **ResU-Net** (Residual U-Net) neural network, trained end-to-end to remove noise and blur from RGB images.

### Project structure
```text
Google Drive (My Drive)/
└── Computational Imaging/               # Root directory of your project on Drive
    │
    ├── dataset/                         # Cached dataset generated by notebook number 0
    │
    ├── notebooks/                       # Core workspace folder for Jupyter Notebooks
    │   └── 1.EndToEndModel.ipynb        # Main training, optimization, and evaluation script
    │
    ├── IPPy/                            # Custom processing library package
    │   ├── IPPy/                        # Core library modules and source code
    │   └── utilities/                   # Helper functions and processing scripts
    │
    └── end_to_end_output/               # Centralized persistent runtime outputs directory
        ├── loss_history/                # Training tracking metrics
        │   └── history.json             # Serialized dictionary containing train and val losses
        │
        ├── metrics_data/                # Quantitative evaluation results
        │   └── (SSIM and PSNR evaluation .csv files and plotted metric charts)
        │
        ├── reconstruction_examples/     # Qualitative visual assessment outputs
        │   └── (Saved image comparisons of degraded vs restored vs clean samples)
        │
        └── weights/                     # Model checkpoint preservation
            └── (Saved state dictionary file containing the best model weights)
```

The workflow covers: data loading, architecture definition, training, metric evaluation, and visual assessment.

The notebook is structured into seven distinct sections:
1. Setup and Inizialization
2. Loading data from the dataset
3. Sanity Check
4. ResU-Net architecture
5. Training Loop
6. Metric-based evaluation
7. Visual evaluation

### 1. Setup and Initialization

#### Execution Environment
The notebook is designed to run on Google Colab, utilizing Google Drive as persistent storage for datasets, weights, and results. The drive is automatically mounted if it is not already present.

The execution of the entire pipeline was performed using Google Colab's T4 GPU.
#### Folder structure

| Variable             | Path                                     | Pourpose                                                 |
| -------------------- | ---------------------------------------- | -------------------------------------------------------- |
| `BASE_DIR`           | `MyDrive/Computational Imaging`          | Project root                                             |
| `DATASET_DIR`        | `.../BASE_DIR/dataset`                   | Directory containing the dataset                         |
| `RESULT_DIR`         | `.../BASE_DIR/end_to_end_output`         | Root directory of the generated output                   |
| `LOSS_HISTORY_DIR`   | `.../RESULT_DIR/loss_history`            | Saves the loss history during training.                  |
| `METRICS_DIR`        | `.../RESULT_DIR/metrics_data`            | Contains CSV files and plots                             |
| `WEIGHTS_DIR`        | `.../RESULT_DIR/weights`                 | Contains the model weights                               |
| `RECONSTRUCTION_DIR` | `.../RESULT_DIR/reconstruction_examples` | Contains images reconstructed by the newly trained model |
 
#### Main libraries/imports

- **PyTorch (`torch`, `torch.nn`, `torch.optim`)** — Deep learning framework used to define and train the network.
- **torchvision** — Image transformations (`ToTensor`).
- **HuggingFace datasets** — Dataset management in Arrow format.
- **IPPy (`psnr`, `ssim`)** — Image quality evaluation metrics.
- **NumPy / Pandas** — Numerical manipulation and results collection.
- **Matplotlib / tqdm** — Progress bar visualization and plotting.

#### Hyperparameters

| Parameter                 | Value | Notes                                                                                                                         |
| ------------------------- | ----- | ----------------------------------------------------------------------------------------------------------------------------- |
| `BATCH_SIZE`              | 8     | Balance between training speed and GPU memory usage                                                                           |
| `EPOCH_NUMBER`            | 30    | Sufficient training epochs to achieve convergence                                                                             |
| `LEARNING_RATE`           | 1e-3  | Standard baseline value for the AdamW optimizer                                                                               |
| `PATIENCE_EARLY_STOPPING` | 7     | given that patience in the scheduler is equal to 3, by setting this parameter equal to 7 we give time to observe improvements |

### 2. Loading data from the dataset

The dataset is saved on disk in HuggingFace Arrow format and is already split into three subsets: **train**, **test**, and **validation**. Each sample contains:
- **x** — The original clean image (ground truth).
- **y_005, y_010, y_050, y_100** — Degraded versions of the image, featuring four increasing levels of noise/blur.

>[!info]
>The degraded images are pre-calculated using a unique, deterministic seed for each image, rather than being generated on-the-fly during training. This ensures that all models within the project are evaluated on identical data (a fairness requirement for comparison) and reduces the computational load during execution. This approach does not cause overfitting due to the vast semantic variety of the dataset and the use of different noise patterns for each image.

Instead of training separate models for each noise level, this class performs a virtual expansion of the sample space by combining each clean image with all of its degraded variants within the same optimization cycle.

The `__len__` function defines the virtual size of the dataset:

$$\text{Virtual Length} = |\text{Original Dataset}| \times |\text{Noise Levels}|$$

The `__getitem__` function operates as follows: when the `DataLoader` requests an item using a sequential linear index $idx \in [0, N-1]$, the class maps the one-dimensional index into a two-dimensional coordinate $(\text{Image}, \text{Noise})$:

- **Image ($idx \mathbin{//} 4$):** The integer division collapses the index into constant blocks of 4 iterations. This ensures that the same clean structural image ($x$) is reused as the target for 4 consecutive optimizer steps.
    
- **Noise ($idx \mathbin{\%} 4$):** The modulo operator (the remainder of the division) continuously cycles through the $[0, 3]$ range. This dynamically selects a different noise column at each iteration.

Example:
```
idx = 0 ==> Img 0, Noise y_005 ──> [Convolutional newtork] ──> Loss vs Target 0 
idx = 1 ==> Img 0, Noise y_010 ──> [Convolutional newtork] ──> Loss vs Target 0 
idx = 2 ==> Img 0, Noise y_050 ──> [Convolutional newtork] ──> Loss vs Target 0 
idx = 3 ==> Img 0, Noise y_100 ──> [Convolutional newtork] ──> Loss vs Target 0 
idx = 4 ==> Img 1, Noise y_005 ──> [Convolutional newtork] ──> Loss vs Target 1
```
#### DataLoader Configuration

| Loader         | Shuffle | Batch size | Note                                     |
| -------------- | ------- | ---------- | ---------------------------------------- |
| `train_loader` | Yes     | 8          | `drop_last=True` for uniform batches     |
| `val_loader`   | No      | 8          | Stable evaluation                        |
| `test_loader`  | No      | 1          | One image at a time for metric precision |

- **`shuffle = True`:** At the beginning of each epoch, the order of the virtual indices is completely reshuffled. Every minibatch will be different across runs, which is crucial to prevent overfitting.
    
- **`shuffle = False`:** Changing the sequence order is not necessary for validation and testing phases.
    
- **`batch_size = 1`:** Standard configuration for testing and final inference. Processing one image at a time allows for calculating the PSNR and SSIM metric vectors with absolute accuracy.

### 3. Sanity Check

An optional verification step that extracts a single batch from the `train_loader` and visualizes the (ground truth, degraded image) pairs for the first 4 images. This serves to confirm that the `DataLoader` is functioning correctly and that the alignments between the clean and degraded images are properly matched.

### 4. ResU-Net Architecture

The ResU-Net combines two established deep learning paradigms:

- **U-Net:** An encoder-decoder architecture with skip connections. Its "funnel" structure captures features at multiple spatial scales.
- **Residual Blocks:** ResNet-inspired residual blocks that allow the gradient to flow directly through the network. This facilitates the training of deep networks and mitigates the vanishing gradient problem.

#### Componenti principali: 

##### ResidualBlock

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        if in_channels != out_channels: 
          self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)

            )
        else:
          self.shortcut = nn.Identity()


    def forward(self, x): 
        shortcut = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += shortcut
        out = self.relu(out)

        return out
```



The foundational building block of the network. Structure:

- `Conv2d(3x3)` $\rightarrow$ `BatchNorm2d` $\rightarrow$ `ReLU`  
- `Conv2d(3x3)` $\rightarrow$ `BatchNorm2d`
- **`Shortcut connection`** $+$ `final` `ReLU`


If `in_channels != out_channels`, the shortcut connection utilizes a `Conv2d(1x1)` with `BatchNorm2d` to match dimensions (**projection shortcut**). Otherwise, it defaults to `nn.Identity()` (**identity shortcut**). The bias is disabled (`bias=False`) in all convolutional layers because the subsequent `BatchNorm2d` renders it redundant.
##### DownBlock (Encoder)

``` python
class DownBlock(nn.Module):
      def __init__(self, in_channels, out_channels, block_type = ResidualBlock):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = block_type(in_channels, out_channels)

      def forward(self, x):
        x = self.pool(x)
        x = self.block(x)
        return x
```

Reduces the spatial resolution by a factor of 2 using `MaxPool2d(2)`, followed by a `ResidualBlock`. The max-pooling operation preserves the most salient features.

##### UpBlock (Decoder)

```python
class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, block_type = ResidualBlock):=
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.block = block_type(out_ch + skip_ch, out_ch)

  

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)

        x = torch.cat([skip, x], dim=1)

        return self.block(x)
```

- `ConvTranspose2d(2x2, stride=2)` — Learnable upsampling (preferred over fixed bilinear upsampling).
    
- **Dimensional alignment** using `F.interpolate` (bilinear) if necessary — handles edge cases involving odd dimensions.
    
- **Concatenation** with the skip connection from the corresponding encoder stage.
    
- `ResidualBlock` to refine the fused features.


#### Struttura completa di `ResUNet`

Con `base_channels = 32`:

```python 
self.initial_block = ResidualBlock(3, 32) 
self.down_block1 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(32, 64)) self.down_block2 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(64, 128)) self.down_block3 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(128, 256))  
self.bottleneck_block = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(256, 512)) 

self.up_block1 = UpBlock(512, 256) 

self.up_block2 = UpBlock(256, 128)
self.up_block3 = UpBlock(128, 64) 
self.up_block4 = UpBlock(64, 32) 
self.final_block = nn.Sequential(nn.Conv2d(32, 3, kernel_size=1), nn.Sigmoid())
```

| Level              | Module                       | # output channels | Resolution  |
| ------------------ | ---------------------------- | ----------------- | ----------- |
| Input              | —                            | 3 ch              | H × W       |
| `initial_block`    | ResidualBlock                | 32                | H × W       |
| `down_block1`      | MaxPool + ResBlock           | 64                | H/2 × W/2   |
| `down_block2`      | MaxPool + ResBlock           | 128               | H/4 × W/4   |
| `down_block3`      | MaxPool + ResBlock           | 256               | H/8 × W/8   |
| `bottleneck_block` | MaxPool + ResBlock           | 512               | H/16 × W/16 |
| `up_block1`        | ConvT + skip(256) + ResBlock | 256               | H/8 × W/8   |
| `up_block2`        | ConvT + skip(128) + ResBlock | 128               | H/4 × W/4   |
| `up_block3`        | ConvT + skip(64) + ResBlock  | 64                | H/2 × W/2   |
| `up_block4`        | ConvT + skip(32) + ResBlock  | 32                | H × W       |
| `final_block`      | Conv2d(1×1) + Sigmoid        | 3                 | H × W       |

The final `Conv2d(1x1)` acts as a "projector" that reduces the 32 channels down to 3 (RGB) without altering the spatial resolution. A `Sigmoid` activation function ensures that the output values are bounded within the $[0, 1]$ range, maintaining consistency with the input normalization.


### 6. Training Loop

#### Optimizer: AdamW
`AdamW` is a variant of the Adam optimizer featuring decoupled weight decay (separated from the adaptive gradient update). It is preferred over standard Adam due to its superior regularization performance and better generalization capabilities.
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
```
#### Scheduler: ReduceLROnPlateau
The learning rate is halved (`factor=0.5`) if the training loss does not improve for 5 consecutive epochs (`patience=3`). This prevents oscillations during the convergence phase without requiring manual scheduling.
```python
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
```
#### Loss: MSELoss
Mean Squared Error is the standard loss function for image restoration tasks. It minimizes the pixel-by-pixel L2 distance between the predicted output and the ground truth. It is fully differentiable and well-defined for continuous normalized values.
``` python
loss_fn = nn.MSELoss()
```

#### Flow per Epoch:

1. **Forward Pass:** The batch of degraded images is fed into the network, which generates the predicted reconstruction.
    
2. **Loss Calculation:** The MSE loss is calculated between the prediction and the clean ground truth.
    
3. **Backward Pass:** `optimizer.zero_grad()` clears the accumulated gradients, `loss.backward()` computes the gradients, and `optimizer.step()` updates the model weights.
    
4. **Logging:** The `tqdm` progress bar displays both the instantaneous loss and the average loss for the current epoch.
    
5. **Scheduler Step:** At the end of each epoch, the learning rate scheduler updates the learning rate based on the average training loss.

``` python
if avg_val_loss < best_val_loss:

   best_val_loss = avg_val_loss
   torch.save(model.state_dict(), w_path)
   epochs_no_improve = 0
   tqdm.write(f"Epoch {epoch+1}/{num_epochs} -> Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} [BEST MODEL SAVED]")

else:
    epochs_no_improve += 1
    tqdm.write(f"Epoch {epoch+1}/{num_epochs} -> Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} [No improvement since{epochs_no_improve} epochs]")
```

``` python
if epochs_no_improve >= patience_early_stopping:

    tqdm.write(f"\n Early Stopping activated. No improvement on validation loss since {patience_early_stopping} epochs.")
	print("Training finished")

     break
```

Upon completion, the model is reloaded using `load_state_dict` and switched to `.eval()` mode for evaluation. This pattern ensures that the testing phase always utilizes the weights that were successfully persisted to disk.

The initialization of `torch.manual_seed(0)` prior to training guarantees the reproducibility of the weight initialization.


### 7. Metric evaluation

#### Metrics Used

- **PSNR (Peak Signal-to-Noise Ratio):** 
    Measures the ratio between the maximum possible power of a signal and the corrupting noise affecting its representation. Expressed in decibels (dB), higher values indicate better reconstruction quality. Formula:
    $$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}^2}{\text{MSE}}\right)$$
    By setting `data_range=1.0`, it assumes that pixel values are normalized within the $[0, 1]$ range.
    
- **SSIM (Structural Similarity Index Measure):**
    
    Measures the perceptual similarity between two images by considering three key components: luminance, contrast, and structure. Its value ranges from -1 to 1, where 1 indicates identical images. It correlates much better with human visual perception compared to PSNR. The argument `channel_axis=2` handles RGB images by processing the channels separately, wrapped in a `try/except` block to ensure compatibility across different versions of `scikit-image`.


### 8. Visual Evaluation

The visual assessment displays a $2 \times 4$ grid for a selected sample from the test set:

- **Row 0:** The 4 degraded images (inputs at the 4 different noise levels).
- **Row 1:** The corresponding reconstructions produced by the network (outputs).
- **Separate Figure:** The clean ground truth image.
    
This setup enables a qualitative inspection of the results, showing how effectively the network recovers details, edges, and colors compared to the original image as a function of the degradation level.

---


## 3.1 Data Degradation (recap)

The dataset is produced by `3_1_Data_Degradation_64_x_64.ipynb`, exactly as in the other pipelines.
A subset of ImageNet-1K classes is loaded and a forward degradation model is applied:

**Only difference from the End-to-End pipeline:** images are downscaled to **64 × 64**
 instead of 256 × 256. A smaller resolution keeps GAN training stable and fast enough to
 fit the time budget, which matters far more for adversarial training than for a
 supervised U-Net.

---

### Project structure

```text
Google Drive (My Drive)/
└── Computational Imaging/               # Project root on Drive
    │
    ├── dataset_64x64/                    # 64x64 dataset generated by notebook 3_1_Data_Degradation_64_x_64
    │
    ├── Homeworks/
    │   ├── 3_2_GAN.ipynb                   # Train the WGAN-GP prior
    │   └── 3_3_GAN_DGP_Latent_Optimization.ipynb   # DGP reconstruction
    │
    ├── IPPy/                             # Custom processing library (IPPy/IPPy/...)
    │
    └── gan_output/                       # All GAN / DGP outputs
        ├── weights/                      # GAN checkpoints (written by file 3_2, read by file 3_3)
        │   └── checkpoint_last.pt        # generator + critic + EMA + optimizers + history
        ├── samples/                      # EMA sample grids saved during training
        ├── metrics_data/                 # training-curve and data-loss images
        └── reconstruction_examples/      # DGP reconstruction figures + the 2x4 input/output grid
```

#### Execution environment

| Variable      | Path                                       | Purpose                                  |
| ------------- | ------------------------------------------ | ---------------------------------------- |
| `BASE_DIR`    | `MyDrive/Computational Imaging`            | Project root                             |
| `DRIVE_DATASET_DIR` / `DATASET_DIR` | `.../BASE_DIR/dataset_64x64` | 64×64 dataset                            |
| `OUTPUT_DIR`  | `.../BASE_DIR/gan_output`                  | Root of all GAN/DGP outputs              |
| `WEIGHTS_DIR` | `.../gan_output/weights`                   | GAN checkpoints                          |
| `SAMPLES_DIR` | `.../gan_output/samples`                   | Sample grids during training             |
| `METRICS_DIR` | `.../gan_output/metrics_data`              | Loss / training-curve images             |
| `RECON_DIR`   | `.../gan_output/reconstruction_examples`   | DGP reconstruction figures               |

---

## 3.2 GAN — WGAN-GP prior (`3_2_GAN.ipynb`)

The goal is to learn a generator `G(z)` that maps a latent vector `z ~ N(0, I)` to a clean
64×64 RGB image, so that the set of images `G` can produce covers the distribution of the
training data. This trained generator becomes the **prior** used in notebook `3_3_GAN_DGP_Latent_Optimization`

The notebook is organized as: setup → DiffAugment → dataset → model → WGAN-GP utilities →
training loop → run → sample inspection → training curves.

#### 3.2.1 Design choices (and why)

| Choice | Why |
| ------ | --- |
| **WGAN-GP** (Wasserstein loss + gradient penalty) | Stable adversarial training; the critic outputs a real-valued score and the gradient penalty enforces the 1-Lipschitz constraint without spectral normalization. |
| **TTUR** (critic LR > generator LR, 1 critic step per generator step) | Keeps the critic slightly ahead of the generator at low computational cost. |
| **EMA** of the generator weights | An exponential moving average gives much smoother, higher-quality samples than the live generator, and a better-behaved prior for latent optimization. |
| **Online DiffAugment** (color + translation) | With only ~4k images the critic would memorize the fixed real set. A fresh random augmentation every epoch gives effectively unlimited variety — the correct way to "expand" a small dataset (far better than saving a fixed set of augmented copies, which adds no information). |
| **bf16 mixed precision** | ~15% faster per epoch; bf16 needs no gradient scaler. The gradient penalty is kept in fp32 for a clean double-backward. |
| **Late cosine LR decay** | The learning rate stays flat, then eases down near the end of training to settle the weights. |

#### 3.2.2 Dataset loading

`CleanImageDataset` returns `(clean_image, label)` with the image scaled to **[-1, 1]**
(to match the generator's `tanh` output). The **labels are ignored** — the GAN is
unconditional. The only augmentation applied here is a random horizontal flip; color and
translation are handled by DiffAugment in the critic path.

#### 3.2.3 DiffAugment

The same differentiable augmentation is applied to **both** real and fake images before
every critic call:

- **color**: random brightness, saturation and contrast,
- **translation**: random spatial shift (zero-padded).

Because the operations are differentiable, the generator's gradient still flows back
through them. (cutout is deliberately excluded to avoid square artifacts.)

#### 3.2.4 Architecture

The generator and critic are mirror residual networks at 64×64.

###### ResBlockUp (generator)
`skip = 1×1 conv on nearest-upsampled input`, main path `BN → ReLU → upsample → 3×3 conv →
BN → ReLU → 3×3 conv`. Nearest-neighbor upsampling avoids checkerboard artifacts.

###### ResBlockDown (critic)
`skip = 1×1 conv → avg-pool`, main path `LeakyReLU → 3×3 conv → LeakyReLU → 3×3 conv →
avg-pool`. **No normalization** is used in the critic: BatchNorm would break the
per-sample gradient penalty.

###### GoodGenerator `G(z) → image ∈ [-1, 1]`
```python
fc:     z (128)            -> 4*4*512
blocks: ResBlockUp(512,512)  # 4  -> 8
        ResBlockUp(512,256)  # 8  -> 16
        ResBlockUp(256,128)  # 16 -> 32
        ResBlockUp(128, 64)  # 32 -> 64
out:    BN -> ReLU -> Conv2d(64,3) -> tanh
```

###### GoodCritic `D(x) → scalar`
```python
conv_in: Conv2d(3, 64)
blocks:  ResBlockDown(64,128)   # 64 -> 32
         ResBlockDown(128,256)  # 32 -> 16
         ResBlockDown(256,512)  # 16 -> 8
         ResBlockDown(512,512)  # 8  -> 4
out:     LeakyReLU -> flatten -> Linear(4*4*512, 1)
```

Weights are initialized with **Xavier uniform**; an **EMA** copy of the generator
(`decay = 0.9995`) is kept for sampling and checkpoints.

#### 3.2.5 WGAN-GP objective

The critic maximizes the gap between real and fake scores, regularized by the gradient
penalty; the generator maximizes the critic's score on its own samples.

$$\mathcal{L}_D = \underbrace{\mathbb{E}[D(\tilde{x})] - \mathbb{E}[D(x)]}_{\text{Wasserstein term}} + \lambda_{gp}\,\mathbb{E}\big[(\lVert \nabla_{\hat{x}} D(\hat{x}) \rVert_2 - 1)^2\big]$$

$$\mathcal{L}_G = -\,\mathbb{E}[D(G(z))]$$

where `x` is real, `x̃ = G(z)` is fake, and `x̂` is a random interpolation between a real
and a fake image. DiffAugment is applied to the `D(real)` and `D(fake)` calls, while the
gradient penalty is computed on **non-augmented** interpolates in fp32.

#### 3.2.6 Training loop

Per batch: one **critic** update (real vs fake, both DiffAugmented, plus the gradient
penalty) then one **generator** update; the EMA is updated after each generator step.
Forward passes run under bf16 autocast; the gradient penalty stays in fp32.

##### Hyperparameters

| Parameter         | Value          | Notes                                            |
| ----------------- | -------------- | ------------------------------------------------ |
| `latent_dim`      | 128            | size of `z`                                       |
| `base_channels`   | 64             | width multiplier of G and D                       |
| `batch_size`      | 64             |                                                  |
| `n_epochs`        | 350            | ≈ 2 h on an RTX 5070 (raise / `RESUME` for more)  |
| `critic_steps`    | 1              | TTUR (1 critic step per generator step)           |
| `lr_d` / `lr_g`   | 3e-4 / 1e-4    | TTUR: critic faster than generator                |
| Adam betas        | (0.0, 0.9)     | recommended for WGAN-GP                            |
| `gp_lambda`       | 10.0           | gradient-penalty weight                           |
| `ema_decay`       | 0.9995         | EMA of the generator                              |
| `diff_aug_policy` | color,translation | online augmentation                            |
| LR decay          | cosine, start 80%, floor 50% | late decay to settle                |

#### 3.2.7 Outputs

- Checkpoints (`generator`, `critic`, `generator_ema`, optimizers, config, history) →
  `gan_output/weights/checkpoint_last.pt` (+ epoch-tagged copies).
- EMA sample grids every 10 epochs → `gan_output/samples/`.
- Training curves (G loss, critic loss, Wasserstein distance, gradient penalty) →
  `gan_output/metrics_data/gan_training_curves.png`.

## 3.3 DGP — Deep Generative Prior reconstruction (`3_3_GAN_DGP_Latent_Optimization.ipynb`)

Given the frozen trained generator, each degraded measurement `y` is reconstructed by
finding the image **on (or near) the generator's manifold** that best explains it. EMA
generator weights are loaded when present (smoother reconstructions).

#### 3.3.1 Forward operator K

The same Gaussian blur used in notebook `3_1_Data_Degradation_64_x_64` (`IPPy.Blurring`, kernel size 9, σ = 2.0) is
rebuilt and reused as the forward operator, so that reconstruction and degradation are
consistent.

#### 3.3.2 Two-stage reconstruction

##### Stage 1 — Latent optimization (GAN inversion)

The generator is **frozen** and only the latent `z` is optimized so that the blurred
generation matches the measurement:

$$z^\* = \arg\min_z \; \lVert K(G(z)) - y \rVert^2 + \lambda_z \lVert z \rVert^2$$

Optimized with Adam + cosine LR decay, over several random **restarts** (the best one is
kept). On its own this stage is limited: the exact target image is usually **not** in the
generator's range, so the reconstruction stays blurry / slightly wrong.

##### Stage 2 — Generator fine-tuning (the actual DGP step)

The generator is then **relaxed**: a per-image clone of `G` is unfrozen and `z` and the
generator weights `θ` are optimized **jointly**, with a penalty that keeps the weights
close to the trained prior:

$$\min_{z,\,\theta} \; \lVert K(G_\theta(z)) - y \rVert^2 + \lambda_z \lVert z \rVert^2 + \lambda_{reg}\,\frac{1}{N}\sum_i \lVert \theta_i - \theta_i^{0} \rVert^2$$

This lets the generator leave its original manifold and adapt to the specific image — which
pure latent search cannot do — **without** degenerating into an unconstrained decoder (the
stay-close penalty + a small number of steps act as the prior / early stopping). The clone
is kept in `eval()` so BatchNorm uses its running statistics (the batch size is 1), and a
fresh clone is used per image so reconstructions are independent.

> The loss uses **only** the measurement `y` and the operator `K` — the ground truth `x` is
> used solely to report PSNR/SSIM, never in the optimization.

##### Hyperparameters

| Parameter        | Value | Notes                                           |
| ---------------- | ----- | ----------------------------------------------- |
| `steps`          | 800   | stage-1 latent optimization steps               |
| `lr`             | 0.02  | stage-1 Adam LR on `z`                          |
| `restarts`       | 2     | stage-1 random restarts (best kept)             |
| `z_l2_weight`    | 1e-4  | weak prior pulling `z` to the origin            |
| `dgp_finetune`   | True  | enable stage 2                                  |
| `dgp_steps`      | 600   | stage-2 joint (z, θ) steps                       |
| `dgp_lr_z`       | 1e-3  | stage-2 LR on `z`                               |
| `dgp_lr_g`       | 2e-4  | stage-2 LR on the generator weights             |
| `dgp_weight_reg` | 0.05  | stay-close-to-prior strength                    |

#### 3.3.3 Metrics

Reconstruction quality is measured with **PSNR** (dB, higher is better) and **SSIM**
(structural similarity, 0–1) from `scikit-image`, computed against the clean image `x`.
The fine-tuning step typically improves PSNR by **several dB** over latent optimization
alone (e.g. ~18–19 dB → ~22–25 dB depending on the noise level).

#### 2.4 Visualization & outputs

- **Per-image figure** (`plot_reconstruction`): 5 panels — `x true`, noisy `y`, `G(z*)`,
  `K(G(z*))`, and the residual `|K(G(z*)) - y|` — saved to
  `gan_output/reconstruction_examples/`.
- **Data-loss curves** (`plot_history`): `||K(G(z)) - y||²` vs step, saved to
  `gan_output/metrics_data/`.
- **2×4 input/output grid** (final cell): for one image, the **top row** shows the degraded
  inputs at the four noise levels (`y_005 … y_100`) and the **bottom row** the corresponding
  DGP reconstructions (annotated with PSNR), with the ground truth shown separately. Saved
  to `gan_output/reconstruction_examples/`.

---

## How to run

1. Run `3_1_Data_Degradation_64_x_64.ipynb` once to build `dataset_64x64/` on Drive.
2. Run `3_2_GAN.ipynb` (`RUN_TRAINING = True`) to train the prior → writes
   `gan_output/weights/checkpoint_last.pt`. Set `RESUME = True` to continue a previous run.
3. Run `3_3_GAN_DGP_Latent_Optimization.ipynb` to reconstruct test images and produce the
   metrics and figures.

### References

Each method used in the two notebooks, with its source:

- **WGAN-GP** (gradient penalty) — I. Gulrajani et al., *Improved Training of Wasserstein GANs*, NeurIPS 2017.
- **DiffAugment** (online differentiable augmentation) — S. Zhao et al., *Differentiable Augmentation for Data-Efficient GAN Training*, NeurIPS 2020.
- **TTUR** (two time-scale update, separate G/D learning rates) — M. Heusel et al., *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium*, NeurIPS 2017.
- **Xavier/Glorot initialization** — X. Glorot, Y. Bengio, *Understanding the difficulty of training deep feedforward neural networks*, AISTATS 2010.
- **Mixed-precision training (bf16)** — P. Micikevicius et al., *Mixed Precision Training*, ICLR 2018.
- **ImageNet dataset** — J. Deng et al., *ImageNet: A Large-Scale Hierarchical Image Database*, CVPR 2009.

---

## 4. Hybrid PD Net + Total Variation

This notebook (`4_Hybrid_PD_Net.ipynb`) implements a hybrid method combining a **Primal-Dual Net (PD-Net)** with **Total Variation (TV)** regularization for image deblurring and denoising operations.

The notebook is structured to train and evaluate the network using pre-computed degraded inputs to ensure fair comparisons across models.

### Key Sections

1. **Configuration and Hyperparameters**
   - **`BATCH_SIZE`**: 4
   - **`EPOCH_NUMBER`**: 30
   - **`LEARNING_RATE`**: 1e-4
   - **`NUM_ITERATIONS`**: 7 (PD-Net iterations)
   - **`CNN_FEATURES`**: 32
   
2. **Data Loading and Dataset Expansion**
   The dataset is loaded from a HuggingFace Arrow format. A custom `DegradedDataset` class performs a virtual expansion of the dataset, associating each clean ground truth image with its multiple degraded versions (`y_005`, `y_010`, `y_050`, `y_100`) dynamically during the training process.

3. **Model Factory**
   Builds the `HybridLearnedPrimalDualTV` model leveraging custom `IPPy` operators. It initializes the convolutional layers using Xavier initialization to maintain training stability.

4. **Training Loop**
   - **Optimizer**: `AdamW`
   - **Scheduler**: `ReduceLROnPlateau` (halves the LR when validation loss plateaus)
   - **Loss Function**: `MSELoss`
   - **Early Stopping**: Halts training if no improvements are seen on the validation loss for `PATIENCE_EARLY_STOPPING` (7) epochs.

5. **Model Evaluation**
   Evaluates the restored images across the 4 noise levels using:
   - **PSNR (Peak Signal-to-Noise Ratio)**
   - **SSIM (Structural Similarity Index)**
   
   The results are aggregated, averaged, and saved into a CSV file (`hybrid_metrics_per_noise_level.csv`).

6. **Image Results and Plots**
   - **Learning Curves**: Generates comprehensive plots showing Train vs Validation Loss (MSE) and PSNR across epochs.
   - **Metrics Bar Charts**: Visualizes the mean PSNR and SSIM scores across different noise levels on the test set.
   - **Visual Reconstruction Grid**: Creates a 2x4 image grid comparing the degraded inputs against the network's outputs for a selected test image, alongside the clean ground truth.

### Setup and Requirements
Designed to run on Google Colab with Google Drive mounted. Requires standard deep learning libraries (`torch`, `torchvision`), `datasets`, and the custom `IPPy` library.
