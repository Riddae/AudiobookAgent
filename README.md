AudiobookAgent Setup Guide

1. Clone the AudiobookAgent Repository
First, clone this AudiobookAgent repository and navigate into its directory.
git clone https://github.com/YOUR_USERNAME/AudiobookAgent.git  # Replace with the actual URL if this is not the main repo
cd AudiobookAgent
Use code with caution.
Bash
2. Create Conda Environment
Create a new Conda environment with Python 3.10 and install pynini.
conda create -n audiobook-env -y python=3.10
conda activate audiobook-env
conda install -y -c conda-forge pynini==2.1.5
Use code with caution.
Bash
3. Clone Sub-Repositories
Next, clone the required external repositories (MMAudio and CosyVoice) into their respective locations.
# Clone MMAudio
git clone https://github.com/hkchengrex/MMAudio.git

# Clone CosyVoice into the TTS directory
# Ensure the 'TTS' directory exists or create it if necessary
mkdir -p TTS # This command ensures 'TTS' directory exists
cd TTS
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
cd .. # Go back to the AudiobookAgent root directory
Use code with caution.
Bash
4. Install Python Dependencies
Install the Python packages for MMAudio and CosyVoice.
# Install MMAudio dependencies
cd MMAudio
pip install -e .
cd .. # Go back to the AudiobookAgent root directory

# Install CosyVoice dependencies
cd TTS/Cosyvoice
pip install -r requirements.txt
cd ../.. # Go back to the AudiobookAgent root directory
Use code with caution.
Bash
5. Download and Setup Models
Download the pre-trained models required for CosyVoice and perform any specific setup for them.
# Create directory for pretrained models
mkdir -p pretrained_models

# Clone CosyVoice2 model
git clone https://www.modelscope.cn/iic/CosyVoice2-0.5B.git pretrained_models/CosyVoice2-0.5B

# Clone CosyVoice-ttsfrd model
git clone https://www.modelscope.cn/iic/CosyVoice-ttsfrd.git pretrained_models/CosyVoice-ttsfrd

# Navigate into the ttsfrd model directory for specific setup
cd pretrained_models/CosyVoice-ttsfrd/

# Unzip resources
unzip resource.zip -d .

# Install ttsfrd specific dependencies (wheel files)
pip install ttsfrd_dependency-0.1-py3-none-any.whl
pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl

# Go back to the AudiobookAgent root directory
cd ../..
Use code with caution.
Bash
Usage
(Coming Soon)
Once all the setup steps are complete, you will be ready to use the AudiobookAgent project. Further instructions on how to run and utilize the various components (e.g., text processing, audio generation, agent logic) will be provided here.
