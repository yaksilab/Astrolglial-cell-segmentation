# Advanced Usage Guide - Astroglial Cell Segmentation

This document provides detailed information about the advanced features and customization options available in the astroglial cell segmentation pipeline.

## Table of Contents
- [Channel Selection](#channel-selection)
- [Image Type Selection](#image-type-selection)
- [Pipeline Modes](#pipeline-modes)
- [Output Structure](#output-structure)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)

## Channel Selection

The pipeline supports independent channel selection for segmentation and extraction, allowing for flexible multi-channel analysis.

### Segmentation Channel
Controls which channel's data is used for the segmentation step:

```bash
# Use channel 1 for segmentation (default)
python -m astroglial_segmentation <path> --segmentation-channel 1

# Use channel 2 for segmentation
python -m astroglial_segmentation <path> --segmentation-channel 2
```

### Extraction Channel
Controls which channel's fluorescence data is extracted:

```bash
# Extract from channel 1 (default)
python -m astroglial_segmentation <path> --extraction-channel 1

# Extract from channel 2
python -m astroglial_segmentation <path> --extraction-channel 2
```

### Cross-Channel Analysis
You can segment on one channel and extract from another:

```bash
# Segment using channel 1, extract traces from channel 2
python -m astroglial_segmentation <path> --segmentation-channel 1 --extraction-channel 2

# Segment using channel 2, extract traces from channel 1
python -m astroglial_segmentation <path> --segmentation-channel 2 --extraction-channel 1
```

## Image Type Selection

The pipeline supports multiple image types from Suite2p for segmentation:

### Available Image Types

| Image Type | Description | Suite2p Key |
|------------|-------------|-------------|
| `meanImg` | Mean image from channel 1 (default) | `meanImg` |
| `meanImg_chan2` | Mean image from channel 2 | `meanImg_chan2` |
| `meanImgE` | Enhanced mean image | `meanImgE` |
| `max_proj` | Maximum projection image | `max_proj` |

### Usage Examples

```bash
# Use enhanced mean image for better contrast
python -m astroglial_segmentation <path> --image-type meanImgE

# Use maximum projection for capturing full cell extent
python -m astroglial_segmentation <path> --image-type max_proj

# Use channel 2 mean image with channel 2 segmentation
python -m astroglial_segmentation <path> --image-type meanImg_chan2 --segmentation-channel 2
```

### Choosing the Right Image Type

- **`meanImg`**: Best for standard segmentation with good signal-to-noise ratio
- **`meanImg_chan2`**: Use when channel 2 has better cell visibility
- **`meanImgE`**: Enhanced contrast, useful for faint cells
- **`max_proj`**: Captures full cell extent, good for cells with extensive processes

## Pipeline Modes

### Full Pipeline (Default)
Runs both segmentation and extraction:

```bash
python -m astroglial_segmentation <path>
```

### Segmentation Only
Skips extraction, useful for mask generation only:

```bash
python -m astroglial_segmentation <path> --no-extraction
```

### Custom Configurations

```bash
# High-quality segmentation with enhanced images
python -m astroglial_segmentation <path> --image-type meanImgE --segmentation-channel 1

# Multi-channel analysis: segment on enhanced image, extract from both channels
python -m astroglial_segmentation <path> --image-type meanImgE --extraction-channel 1
python -m astroglial_segmentation <path> --image-type meanImgE --extraction-channel 2
```

## Output Structure

### Segmentation Outputs

Files created in the Suite2p output folder:

```
<suite2p_folder>/
├── CP2_s1_039189_<image_type>_ch<channel>.png
├── CP2_s1_039189_<image_type>_ch<channel>_seg.npy
├── CP2_s2_039234_<image_type>_ch<channel>.png
├── CP2_s2_039234_<image_type>_ch<channel>_seg.npy
├── CP2_s3_039234_<image_type>_ch<channel>.png
├── CP2_s3_039234_<image_type>_ch<channel>_seg.npy
└── combined_<image_type>_ch<channel>_seg.npy
```

### Extraction Outputs

Files created in `cellpose_suite2p_output_ch<channel>/`:

```
cellpose_suite2p_output_ch<channel>/
├── F.npy                              # Raw fluorescence (channel 1)
├── F_chan2.npy                        # Raw fluorescence (channel 2, if available)
├── F_processed_ch<channel>.npy        # Processed fluorescence from selected channel
├── Fneu.npy                           # Raw neuropil (channel 1)
├── Fneu_chan2.npy                     # Raw neuropil (channel 2, if available)
├── Fneu_processed_ch<channel>.npy     # Processed neuropil from selected channel
├── iscell.npy                         # Cell classification
├── ops.npy                            # Suite2p parameters
├── spks.npy                           # Deconvolved spikes
└── stat.npy                           # ROI statistics
```

## Troubleshooting

### Common Issues

#### 1. Image Type Not Found
```
KeyError: Image type 'max_proj' not found in ops file
```
**Solution**: Check which image types are available in your Suite2p output:
```python
import numpy as np
ops = np.load('/path/to/ops.npy', allow_pickle=True).item()
available_images = [key for key in ['meanImg', 'meanImg_chan2', 'meanImgE', 'max_proj'] if key in ops]
print("Available image types:", available_images)
```

#### 2. Channel 2 Data Missing
```
Warning: Channel 2 data file not found, falling back to channel 1
```
**Solution**: This is normal if your data only has one channel. The pipeline automatically falls back to channel 1.

#### 3. No Mask Files Found
```
FileNotFoundError: One or more mask files not found
```
**Solution**: Ensure segmentation completed successfully. Check for `*_seg.npy` files in the output folder.

## API Reference

### Main Function
```python
def main_pipeline(data_path, segmentation_channel=1, extraction_channel=1, 
                 image_type="meanImg", run_extraction=True):
    """
    Main pipeline for astroglial cell segmentation and trace extraction.
    
    Args:
        data_path (str): Path to Suite2p output folder
        segmentation_channel (int): Channel to use for segmentation (1 or 2)
        extraction_channel (int): Channel to use for extraction (1 or 2)
        image_type (str): Type of image to use for segmentation 
        run_extraction (bool): Whether to run extraction after segmentation
    """
```

### Segmentation Function
```python
def segment_cells(data_path, segmentation_channel=1, image_type="meanImg", model_dirs=model_dirs):
    """
    Perform segmentation using multiple Cellpose models and combine the results.
    
    Args:
        data_path (str): Path to Suite2p output folder
        segmentation_channel (int): Channel to use for segmentation (1 or 2)
        image_type (str): Type of image to use for segmentation 
        model_dirs (list): List of model directories to use for segmentation
    """
```

### Extraction Function
```python
def create_suite2p_masks_extract_traces(working_dir, extraction_channel=1, cp_seg_file=None):
    """
    Create Suite2p compatible masks and extract traces.
    
    Args:
        working_dir (str): Path to Suite2p output folder
        extraction_channel (int): Channel to use for extraction (1 or 2)
        cp_seg_file (str, optional): Specific segmentation file to use
    """
```

## Best Practices

### 1. Workflow Recommendations
1. First run segmentation only (`--no-extraction`) to check mask quality
2. Adjust parameters if needed and re-run
3. Run full pipeline with optimized parameters
4. Compare results across different configurations if needed