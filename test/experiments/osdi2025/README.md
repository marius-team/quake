# OSDI 2025 Artifact

This directory contains the experiments for Quake's OSDI 2025 submission. 




### Installation

**Requirements:**
- Python 3.9 or later
- Linux (Tested on Ubuntu 22.04) 

1. **Clone the Repository:**

```bash
git clone https://github.com/marius-team/quake.git
cd quake
git submodule update --init --recursive
```

2. **Set Up the Conda Environment:**

   Create and activate the environment using the provided YAML file:

```bash
conda env create -f environments/ubuntu-latest/conda.yaml
conda activate quake-env
```

3. **Install Quake:**

```bash
pip install --no-use-pep517 .
```

### Kick the tires

To make sure Quake and the experiment environment are set up correctly, you can run the following commands:

**Make sure to run all commands in the repository root directory**

1. **Run the quickstart example:**

This is a basic test that the main operations of Quake are working.

```bash
python3 examples/quickstart.py
```

2. **Run the Benchmark:**

```bash
python3 -m test.experiments.osdi2025.experiment_runner --experiment kick_the_tires --config sift1m
```

---