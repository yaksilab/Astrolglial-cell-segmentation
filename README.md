# Astrolglial-cell-segmentation

This project aims to segment astroglial cells from mean images generated from motion corrected movie in suite2p. 

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


## Usage
 Activate the environment:

```bash
conda activate astroglial-seg-extraction
```


Run the pipeline on your suite2p output folder:

```bash
python -m astroglial_segmentation.pipeline <path to your suit2p output folder>
```