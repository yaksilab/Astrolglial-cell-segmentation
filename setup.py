# -*- coding: utf-8 -*-
from setuptools import setup

package_dir = \
{'': 'src'}

packages = \
['astroglial_segmentation']

package_data = \
{'': ['*']}

install_requires = \
['cellpose==2.3.2',
 'matplotlib>=3.8.2,<4.0.0',
 'numpy>=1.26.2,<2.0.0',
 'opencv-python>=4.8.1.78,<5.0.0.0',
 'suite2p==0.14.3']

setup_kwargs = {
    'name': 'astroglial_segmentation',
    'version': '0.1.0',
    'description': 'doing cell detection and segmentation of radial glial of zebra fish brain cells',
    'long_description': '# Astrolglial-cell-segmentation\n\nThis project aims to segment astroglial cells from mean images generated from motion corrected movie in suite2p. \n\n## Installation\n\nClone the repository and install the required packages using the following command:\n\n```bash\ngit clone https://github.com/yaksilab/Astrolglial-cell-segmentation.git\n```\n\nOpen Anaconda promt and create an anaconda environment with the projects environment.yml file:\n\n```bash\ncd Astrolglial-cell-segmentation\n```\n\n```bash\nconda env create -f environment.yml\n```\n\nActivate the environment:\n\n```bash\nconda activate Astroglial-segmentation\n```\n\ninstall the dependencies from the projects requirements.txt file:\n\n```bash\npip install -r requirements.txt\n```\n\n\n## Usage\n Activate the environment:\n\n```bash\nconda activate Astroglial-segmentation\n```\n\nGo into the project directory:\n\n```bash\ncd Astrolglial-cell-segmentation\n```\n\nRun the pipeline on your suite2p output folder:\n\n```bash\npython -m src.pipeline <path to your suit2p output folder>\n```',
    'author': 'javidr',
    'author_email': 'javidr@ntnu.no',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'None',
    'package_dir': package_dir,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.9.18,<4.0.0',
}


setup(**setup_kwargs)

