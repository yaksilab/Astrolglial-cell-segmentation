# Astrolglial-cell-segmentation

This project aims to segment astroglial cells from images generated from motion corrected movies in Suite2p using an ensemble of custom-trained Cellpose models. The pipeline provides flexible options for channel selection, image type selection, and optional trace extraction.



## Installation

Create a minimal anaconda environment, you can download an environment file from the repository root:

[environment.yaml](environment.yaml)

Open Anaconda prompt and create an anaconda environment with the project's environment.yaml file:

```bash
conda env create -f environment.yaml
```

Activate the environment:

```bash
conda activate astroglial-segmentation-extraction
```

Install the package from the GitHub repository:

```bash
pip install git+https://github.com/yaksilab/Astrolglial-cell-segmentation.git
```

## Usage

### Basic Usage

Activate the environment:

```bash
conda activate astroglial-segmentation-extraction
```

Run the pipeline on your Suite2p output folder with default settings:

```bash
python -m astroglial_segmentation <path_to_suite2p_output_folder>
```

**Default behavior**: Performs segmentation and extraction on channel 1 using the mean image (`meanImg`).

### Advanced Usage

The pipeline supports various options for customizing segmentation and extraction:

#### Channel Selection

Select different channels for segmentation and extraction:

```bash
# Use channel 2 for segmentation, channel 1 for extraction
python -m astroglial_segmentation <path> --segmentation-channel 2

# Use channel 1 for segmentation, channel 2 for extraction  
python -m astroglial_segmentation <path> --extraction-channel 2

# Use channel 2 for both segmentation and extraction
python -m astroglial_segmentation <path> --segmentation-channel 2 --extraction-channel 2
```

#### Image Type Selection

Choose which image type to use for segmentation:

```bash
# Use channel 2 mean image for segmentation
python -m astroglial_segmentation <path> --image-type meanImg_chan2

# Use enhanced mean image for segmentation
python -m astroglial_segmentation <path> --image-type meanImgE

# Use maximum projection image for segmentation
python -m astroglial_segmentation <path> --image-type max_proj
```

**Available image types:**
- `meanImg` - Mean image from channel 1 (default)
- `meanImg_chan2` - Mean image from channel 2
- `meanImgE` - Enhanced mean image
- `max_proj` - Maximum projection image

#### Segmentation Only

Skip the extraction step and only perform segmentation:

```bash
python -m astroglial_segmentation <path> --no-extraction
```

#### Combined Examples

```bash
# Segment using channel 2 max projection, extract from channel 2
python -m astroglial_segmentation <path> --segmentation-channel 2 --image-type max_proj --extraction-channel 2

# Segment using enhanced mean image, skip extraction
python -m astroglial_segmentation <path> --image-type meanImgE --no-extraction

# Use channel 1 for segmentation with max projection, channel 2 for extraction
python -m astroglial_segmentation <path> --image-type max_proj --extraction-channel 2
```

### Command Line Options

```
positional arguments:
  data_path             Path to Suite2p output folder

optional arguments:
  -h, --help            Show help message and exit
  --segmentation-channel {1,2}
                        Channel to use for segmentation (1 or 2, default: 1)
  --extraction-channel {1,2}
                        Channel to use for extraction (1 or 2, default: 1)
  --image-type {meanImg,meanImg_chan2,meanImgE,max_proj}
                        Type of image to use for segmentation (default: meanImg)
  --no-extraction       Skip extraction step, only perform segmentation
```

## Output

### Segmentation Output

The segmentation step creates the following files in your Suite2p output folder:

- `<model>_<image_type>_ch<channel>.png` - Input images for each model
- `<model>_<image_type>_ch<channel>_seg.npy` - Individual model segmentation results
- `combined_<image_type>_ch<channel>_seg.npy` - Combined segmentation mask

### Extraction Output

The extraction step creates a new folder `cellpose_suite2p_output_ch<channel>` containing:

- `F.npy` - Fluorescence traces (channel 1)
- `F_chan2.npy` - Fluorescence traces (channel 2, if available)
- `F_processed_ch<channel>.npy` - Processed fluorescence from selected channel
- `Fneu.npy` - Neuropil fluorescence (channel 1)
- `Fneu_chan2.npy` - Neuropil fluorescence (channel 2, if available)
- `Fneu_processed_ch<channel>.npy` - Processed neuropil from selected channel
- `iscell.npy` - Cell classification results
- `ops.npy` - Suite2p operations parameters
- `spks.npy` - Deconvolved spike traces
- `stat.npy` - ROI statistics

## Model Ensemble

The pipeline uses three specialized Cellpose models:

1. **CP2_s1_039189** - Complete cell segmentation
2. **CP2_s2_039234** - Cell body segmentation  
3. **CP2_s3_039234** - Cell process segmentation

These models are combined using overlap thresholds to create optimal segmentation results.

## Requirements

- Python 3.9.18
- Suite2p
- Cellpose
- NumPy
- Matplotlib

## Notice

If no options are specified, it will:

- Use channel 1 for both segmentation and extraction
- Use `meanImg` for segmentation
- Run both segmentation and extraction steps
- Look for legacy file naming conventions if new ones are not found