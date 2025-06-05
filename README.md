

````markdown
# AudiobookAgent Setup Guide

This repository helps you set up the environment and dependencies required for running the AudiobookAgent system, which involves cloning relevant repositories, setting up a Conda environment, and downloading pretrained models.

### 1. **Clone the Repositories**

First, clone the necessary repositories for this project:

```bash
# Clone the MMAudio repository
git clone https://github.com/hkchengrex/MMAudio.git
cd MMAudio

# Clone the CosyVoice repository (with submodules)
cd TTS
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
````

### 2. **Create Conda Environment**

Create a new Conda environment for this project:

```bash
# Create a Conda environment with Python 3.10
conda create -n name -y python=3.10

# Activate the environment
conda activate name

# Install pynini version 2.1.5
conda install -y -c conda-forge pynini==2.1.5
```

### 3. **Install Python Dependencies**

Once your environment is set up, install the Python dependencies for the repositories:

```bash
# Install MMAudio dependencies
cd MMAudio
pip install -e .

# Install TTS/CosyVoice dependencies
cd TTS/CosyVoice
pip install -r requirements.txt
```

### 4. **Download Pretrained Models**

Now, let's download the necessary pretrained models:

```bash
# Create a directory for pretrained models
mkdir -p pretrained_models

# Clone the pretrained models for CosyVoice
git clone https://www.modelscope.cn/iic/CosyVoice2-0.5B.git pretrained_models/CosyVoice2-0.5B
git clone https://www.modelscope.cn/iic/CosyVoice-ttsfrd.git pretrained_models/CosyVoice-ttsfrd
```

### 5. **Install Additional Dependencies for the Pretrained Models**

Navigate to the `CosyVoice-ttsfrd` folder and install the additional dependencies:

```bash
cd pretrained_models/CosyVoice-ttsfrd/

# Unzip resources
unzip resource.zip -d .

# Install the ttsfrd dependencies
pip install ttsfrd_dependency-0.1-py3-none-any.whl
pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl
```


