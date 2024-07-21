# data-driven-legged-locomotion
# Building the project
The build procedure has been tested on Ubuntu 22.04.1 LTS. 

## Install the system dependencies
The project depends on some system libraries to build the external dependencies. To install them, run the following command:
```sudo apt-get update && sudo apt-get install cmake libgl1-mesa-dev libxinerama-dev libxcursor-dev libxrandr-dev libxi-dev ninja-build zlib1g-dev clang-12```

## Setup the conda environment
In order to easily install the python dependencies, a conda environment is provided in the `spec-file.txt` file. To create the conda environment, run the following command:
```bash
conda create --name mujoco --file spec-file.txt
```
where `<env_name>` is the name of the conda environment you want to create. To activate the environment, run:
```bash
conda activate mujoco
```

## Install the external dependencies
To install the external dependencies, run the following command inside the conda environment you just created:
```bash
python3 setup.py install
```
This will download, build and install all of the required dependencies.