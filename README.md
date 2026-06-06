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