# M2m
A Pytorch Lightning template for implementing code for the paper "M2m: Imbalanced Classification via Major-to-minor Translation" (CVPR 2020)

## Acknowledgments

This project is based on the ideas and concepts from the following repo:
- [M2m](https://github.com/alinlab/M2m) by [Jaehyung Kim](https://github.com/bbuing9).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/souravraha/M2m.git
   cd M2m
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

To train the model using PyTorch Lightning CLI, follow the steps below:

1. Make sure you have installed the required dependencies as mentioned in the Installation section.

2. Prepare your dataset.

3. Change directory:
   ```bash
   cd src
   ```

4. Edit the config file:
   ```bash
   nano config.yaml
   ```

5. Train the model using the following command:
   ```bash
   python main.py fit -c config.yaml --hyperparameter1 value1 --hyperparameter2 value2 ...
   ```
   The trailining hyperparameter args will override their values in config.yaml file.
   
   Example:
   ```bash
   python main.py fit -c config.yaml --lr 0.001 --batch_size 32
   ```
   overrides the learning rate and batch size specified in config.yaml.

   You can view all available command-line arguments by running:
   ```bash
   python main.py fit --help
   ```
   
## Contributing

We welcome contributions to improve this project. Feel free to open issues and pull requests.
