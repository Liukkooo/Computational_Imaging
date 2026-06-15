# End-to-End Model Documentation

This notebook implements a complete image restoration pipeline based on a **ResU-Net** (Residual U-Net) neural network, trained end-to-end to remove noise and blur from RGB images.

The workflow covers: data loading, architecture definition, training, metric evaluation, and visual assessment.

The notebook is structured into seven distinct sections:
1. Setup and Inizialization
2. Loading data from the dataset
3. Sanity Check
4. ResU-Net architecture
5. Training Loop
6. Metric-based evaluation
7. Visual evaluation

## 1. Setup and Initialization

### Execution Environment
The notebook is designed to run on Google Colab, utilizing Google Drive as persistent storage for datasets, weights, and results. The drive is automatically mounted if it is not already present.

The execution of the entire pipeline was performed using Google Colab's T4 GPU.
### Folder structure

| Variable             | Path                                     | Pourpose                                                 |
| -------------------- | ---------------------------------------- | -------------------------------------------------------- |
| `BASE_DIR`           | `MyDrive/Computational Imaging`          | Project root                                             |
| `DATASET_DIR`        | `.../BASE_DIR/dataset`                   | Directory containing the dataset                         |
| `RESULT_DIR`         | `.../BASE_DIR/end_to_end_output`         | Root directory of the generated output                   |
| `LOSS_HISTORY_DIR`   | `.../RESULT_DIR/loss_history`            | Saves the loss history during training.                  |
| `METRICS_DIR`        | `.../RESULT_DIR/metrics_data`            | Contains CSV files and plots                             |
| `WEIGHTS_DIR`        | `.../RESULT_DIR/weights`                 | Contains the model weights                               |
| `RECONSTRUCTION_DIR` | `.../RESULT_DIR/reconstruction_examples` | Contains images reconstructed by the newly trained model |
 
### Main libraries

- **PyTorch (`torch`, `torch.nn`, `torch.optim`)** — Deep learning framework used to define and train the network.
- **torchvision** — Image transformations (`ToTensor`).
- **HuggingFace datasets** — Dataset management in Arrow format.
- **scikit-image (`psnr`, `ssim`)** — Image quality evaluation metrics.
- **NumPy / Pandas** — Numerical manipulation and results collection.
- **Matplotlib / tqdm** — Progress bar visualization and plotting.

### Hyperparameters

| Parameter                 | Value | Notes                                                                                                                         |
| ------------------------- | ----- | ----------------------------------------------------------------------------------------------------------------------------- |
| `BATCH_SIZE`              | 8     | Balance between training speed and GPU memory usage                                                                           |
| `EPOCH_NUMBER`            | 30    | Sufficient training epochs to achieve convergence                                                                             |
| `LEARNING_RATE`           | 1e-3  | Standard baseline value for the AdamW optimizer                                                                               |
| `PATIENCE_EARLY_STOPPING` | 7     | given that patience in the scheduler is equal to 3, by setting this parameter equal to 7 we give time to observe improvements |

## 2. Loading data from the dataset

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

## 3. Sanity Check

An optional verification step that extracts a single batch from the `train_loader` and visualizes the (ground truth, degraded image) pairs for the first 4 images. This serves to confirm that the `DataLoader` is functioning correctly and that the alignments between the clean and degraded images are properly matched.

## 4. ResU-Net Architecture

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


## 6. Training Loop

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


## 7. Metric evaluation

### #### Metrics Used

- **PSNR (Peak Signal-to-Noise Ratio):** 
    Measures the ratio between the maximum possible power of a signal and the corrupting noise affecting its representation. Expressed in decibels (dB), higher values indicate better reconstruction quality. Formula:
    $$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}^2}{\text{MSE}}\right)$$
    By setting `data_range=1.0`, it assumes that pixel values are normalized within the $[0, 1]$ range.
    
- **SSIM (Structural Similarity Index Measure):**
    
    Measures the perceptual similarity between two images by considering three key components: luminance, contrast, and structure. Its value ranges from -1 to 1, where 1 indicates identical images. It correlates much better with human visual perception compared to PSNR. The argument `channel_axis=2` handles RGB images by processing the channels separately, wrapped in a `try/except` block to ensure compatibility across different versions of `scikit-image`.


## 8. Visual Evaluation

The visual assessment displays a $2 \times 4$ grid for a selected sample from the test set:

- **Row 0:** The 4 degraded images (inputs at the 4 different noise levels).
- **Row 1:** The corresponding reconstructions produced by the network (outputs).
- **Separate Figure:** The clean ground truth image.
    
This setup enables a qualitative inspection of the results, showing how effectively the network recovers details, edges, and colors compared to the original image as a function of the degradation level.