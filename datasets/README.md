# Datasets

This directory is used to store datasets for training and testing.
Please place your hyperspectral image datasets here or configure the paths in the code accordingly.

## Dataset Preprocessing Scripts (`mcodes/` directory)

In each dataset's `mcodes/` folder, you will typically find 4 MATLAB scripts used for data preprocessing and degradation:

1. `generate_[DatasetName].m` (e.g., `generate_Pavia.m`): Reads the original raw open-source HSI dataset, normalizes the spectral values, and splits it into training and testing parts.
2. `generate_train_data.m`: The main entry point to generate the training set. It loops through the split training images and passes them to the cropping function.
3. `crop_image.m`: The core function called by `generate_train_data.m`. It crops large images into smaller patches, applies data augmentation (rotations, flips), and simulates degradation (Gaussian blur followed by downsampling) to produce the `lq`, `bic` (bicubic), and `gt` (ground truth) patch pairs.
4. `generate_test_data.m`: The main entry point to generate the testing set. It loads the split testing images and directly applies the same degradation model to produce full-image testing pairs.

> **Note on the CAVE dataset:** The `CAVE` directory does not contain a specific `generate_CAVE.m` script. To prepare CAVE data, you simply need to normalize the raw images directly and place them into the designated training/testing subdirectories before running `generate_train_data.m` and `generate_test_data.m`.

## Data Generation (`matlab_run.py`)

This directory contains the `matlab_run.py` script, which is used to execute MATLAB scripts (like `generate_test_data.m`) directly from Python to preprocess or generate data.

**Prerequisites:**
- Ensure MATLAB is installed on your system.
- Install the MATLAB Engine API for Python:
  ```bash
  pip install matlabengine
  ```

**Usage:**
1. Open `matlab_run.py`.
2. Update the target MATLAB script path inside `eng.eval(...)` to match your local setup (e.g., `/path/to/generate_test_data.m`).
3. Run the script:
  ```bash
  python matlab_run.py
  ```

## Notes on Changing the Super-Resolution Scale Factor

When switching to a different upsampling scale factor (e.g., x2, x3, x4), please ensure you modify the following:

**Dataset Generation (MATLAB)**: 
Update the `factor` variable (e.g., `factor = 1/4;` for x4) in the respective dataset's `generate_test_data.m` and the training data generation scripts that call `crop_image.m`. The output folder paths will adapt automatically.
