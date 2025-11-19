# Dual-Attention Deep Reinforcement Learning for Cooperative Robot Exploration

Deep reinforcement learning framework for multi-robot cooperative exploration using dual-attention mechanisms and Dueling DQN.

## Related Repositories

- [train_assigner_dqn](https://github.com/Morris73198/train_assigner_dqn) - Training module for assigner DQN
- [robot_rl](https://github.com/Morris73198/robot_rl) - Original robot reinforcement learning repository

## Prerequisites

- Python 3.6
- CUDA-capable GPU (recommended) or CPU
- Conda package manager

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Morris73198/robot_rl
cd robot_rl
```

### 2. Create Conda Environment

```bash
conda create -p ./env3.6 python=3.6
source activate ./env3.6
```

### 3. Install Dependencies

#### With GPU Support

```bash
pip install -e .
pip install tensorflow-gpu==2.6.0
pip install keras==2.6.0
conda install cudnn=8.9.2.26
conda install cudatoolkit==11.8
pip install opencv-python==4.6.0.66
pip install opencv-python tqdm numpy
pip install pandas
```

#### CPU Only (Without GPU)

If you don't have a GPU, you can install TensorFlow for CPU:

```bash
pip install -e .
pip install tensorflow==2.6.0
pip install keras==2.6.0
pip install opencv-python==4.6.0.66
pip install opencv-python tqdm numpy
pip install pandas
```

## Dataset Setup

### 1. Create Data Directory

```bash
mkdir data
```

### 2. Download Dataset

Download the DungeonMaps dataset from Google Drive:

[DungeonMaps Dataset](https://drive.google.com/drive/folders/1Arinxv805GdHE5CKgcFr-cXESRTs94ay?usp=drive_link)

### 3. Extract and Organize

After downloading, organize the files in the following structure:

```
data/
└── DungeonMaps/
    ├── train/
    └── test/
```

## Project Structure

```
.
├── two_robot_dueling_dqn_attention/
│   ├── environment/          # Multi-robot environment implementations
│   ├── models/              # Neural network models and trainers
│   ├── utils/               # Utility functions and C++ extensions
│   ├── config.py            # Configuration file
│   └── visualization.py     # Visualization utilities
├── scripts/
│   └── two_robot_dueling_dqn_attention/
│       ├── train.py         # Training script
│       ├── test.py          # Testing script
│       ├── test5.py         # Additional test script
│       └── test_keras.py    # Keras model testing
├── data/                    # Dataset directory (created during setup)
└── setup.py                 # Package installation script
```

## Usage

### Training

```bash
python scripts/two_robot_dueling_dqn_attention/train.py
```

### Testing

```bash
python scripts/two_robot_dueling_dqn_attention/test.py
```

## Key Features

- **Dual-Attention Mechanism**: Enhances robot coordination and exploration efficiency
- **Dueling DQN Architecture**: Separates state value and action advantage estimation
- **Multi-Robot Coordination**: Supports cooperative exploration with multiple robots
- **Local Map Tracking**: Individual robot local map management
- **Unknown Environment Handling**: Adaptive exploration in unknown environments

## Requirements

See `setup.py` for detailed package requirements. Key dependencies:
- TensorFlow >= 2.0.0
- NumPy >= 1.19.0
- Matplotlib >= 3.3.0
- scikit-image >= 0.17.0
- pybind11 >= 2.6.0
- OpenCV Python
- Pandas

## Troubleshooting

### CUDA/cuDNN Issues

If you encounter CUDA or cuDNN compatibility issues:
1. Verify your GPU driver version
2. Ensure CUDA 11.8 and cuDNN 8.9.2.26 are correctly installed
3. Check TensorFlow-GPU compatibility with your CUDA version

### C++ Extension Build Issues

If the inverse sensor model C++ extension fails to build:
1. Ensure you have a C++11 compatible compiler
2. Verify Eigen3 is installed in one of the expected locations
3. On Windows, install Visual Studio Build Tools

### Import Errors

If you encounter import errors after installation:
```bash
# Reinstall in development mode
pip install -e .
```

## License

Please refer to the repository license file.

## Citation

If you use this code in your research, please cite the original paper.
