# Infernus
A repository for gravitational wave inference, background collection, and injection runs with deep learning models. Specifically built to work with deep learning models that take gravitational wave signal-to-noise (SNR) time series as input.

## Workflow Overview

Broadly, infernus orchestrates the following tasks:

1. Generate training,validation, and test datasets using GWSamplegen.
2. Tune/train a deep learning model using the generated datasets.
3. Convert the trained model(s) to a Triton Inference Server model repository for fast inference.
4. Run background collection with the trained model(s)
5. Inject simulated signals into data and run inference with the trained model(s)
6. Use the results of background and injection runs to calculate false alarm rates and determine the sensitivity of the trained model(s).
7. Perform a targeted search for known events in real data using the trained model(s).
8. Perform a blind search in real data using the trained model(s).

Not all of the tasks have to be completed in the same workflow, for example, if you already have a trained model, you can skip steps 1 and 2. Similarly, you can avoid step 8 if you only want to run a targeted search for known events.

## Installation

This repository has `GWSamplegen` as a dependency. Follow the installation instructions [here](https://github.com/alistair-mcleod/GWSamplegen), Then `cd` out of GWSamplegen and clone this repository with:

```
git clone https://github.com/alistair-mcleod/infernus.git
```
then `cd` into Infernus and run

```
pip install .
```

Finally, you'll need to install a Triton Inference Server. We used version 22.11, and installed it using apptainer.

```
apptainer build tritonserver.sif docker://nvcr.io/nvidia/tritonserver:22.11-py3
```


## Usage 

1. Create a json file using the `share/000config.json` file as a template. This file should contain all parameters necessary for the entire workflow, from dataset generation to background and injection runs. If necessary, specify the tuning/training directories for the model(s) you want to use. If you already have a trained model, you can skip this step. Define the noise segment(s) to use, the template bank to use, and where to save the run files. 
2. Run make_workflow.py with the created json file as a command-line argument. This will make a workflow directory with the necessary scripts to run background and injection runs.
3. Run the desired parts of the workflow using SLURM. The shell script `bin/run_full_workflow.sh` will queue up most of the components of the workflow except for the blind search (step 8). The blind search is not included because it requires a lot of CPU resources, and is best run when models have first been validated.
4. To run a background/injection/p_astro/search job by itself, use `bin/filter_and_predict.sh` with the created json file as a command-line argument.

If a background/injection/p_astro/search job fails due to some of the SLURM job array jobs failing, you can re-run the failed jobs using the `bin/recovery.sh` script. The cleanup log file for the failed job will include a one-liner command for conveniently re-running the failed jobs.



## Configuration file options

All of the options for creating a Triton server model repository, and running background and injection runs are stored in `.json` files.

TODO: Add a full description of the configuration file options here. 


## Citation

If you use this code in your research, please cite the following paper:

```
@article{McLeod2025,
    title = "{Binary neutron star merger search pipeline powered by deep learning}",
    author = "McLeod, Alistair and Beveridge, Damon and Wen, Linqing and Wicenec, Andreas",
    journal = {Phys. Rev. D},
    volume = {111},
    number = {2},
    pages = {024035},
    year = {2025},
	month = {Jan},
    eprint = {2409.06266},
    archivePrefix = {arXiv},
    primaryClass = {gr-qc},
    doi = {10.1103/PhysRevD.111.024035},
	url = {https://doi.org/10.1103/PhysRevD.111.024035}
}
