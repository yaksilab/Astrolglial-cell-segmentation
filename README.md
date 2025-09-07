# Astrolglial-cell-segmentation

This project aims to segment astroglial cells from mean images generated from motion corrected movie in suite2p. 

More documentation can be found in the [docs](docs\README.md).

## Installation


Open Anaconda promt and create an minimal conda environment with the projects environment.yml file:

```bash
conda create --name astroglial-seg-extraction python=3.9
```

Activate the environment:

```bash
conda activate astroglial-seg-extraction
```

install the project from this repository:

```bash
pip install git+https://github.com/yaksilab/Astrolglial-cell-segmentation.git
```


## Basic Usage

Activate the environment:

```bash
conda activate astroglial-seg-extraction
```

Run the pipeline on your suite2p output folder:

```bash
python -m astroglial_segmentation <path to your suit2p output folder>
```

### Usage via API

You can also use the functions in the package directly in your scripts or Jupyter notebooks. Here is an example:

First activate the environment:

```bash
conda activate astroglial-seg-extraction
```

Then in your Python script or Jupyter notebook, you can do:

```python
from astroglial_segmentation import segment_cells, create_suite2p_masks_extract_traces
data_path = "<path to your suite2p output folder>"
segment_cells(data_path)
create_suite2p_masks_extract_traces(data_path)
```