# Astrolglial-cell-segmentation

This project aims to segment astroglial cells from mean images generated from motion corrected movie in suite2p. 

## Installation

Clone the repository and install the required packages using the following command:

```bash
git clone https://github.com/yaksilab/Astrolglial-cell-segmentation.git
```

Open Anaconda promt and create an anaconda environment with the projects environment.yml file:

```bash
cd Astrolglial-cell-segmentation
```

```bash
conda env create -f environment.yaml
```

Activate the environment:

```bash
conda activate Astroglial-segmentation
```

install the dependencies from the projects requirements.txt file:

```bash
pip install -r requirements.txt
```

install the project as a package in your conda environment:

```bash
pip install -e .
```




## Usage
 Activate the environment:

```bash
conda activate Astroglial-segmentation
```

Go into the project directory:

```bash
cd Astrolglial-cell-segmentation
```

Run the pipeline on your suite2p output folder:

```bash
python -m astroglial_segmentation.pipeline <path to your suit2p output folder>
```