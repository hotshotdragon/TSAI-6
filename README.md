# MNIST Model Training

![Build Status](https://img.shields.io/github/actions/workflow/status/yourusername/yourrepo/model_checks.yml?branch=main)
![Python Version](https://img.shields.io/badge/python-3.8-blue)

This project contains a PyTorch implementation of a convolutional neural network (CNN) for classifying the MNIST dataset. The model is defined in a modular format, and GitHub Actions are used to ensure the model architecture meets specific requirements.

## Project Structure

- `models/mnist_net.py`: Contains the model definition.
- `utils/training.py`: Contains utility functions for training and testing the model.
- `train.py`: Main script to train and test the model.
- `.github/workflows/model_checks.yml`: GitHub Actions workflow to check model architecture.
- `tests/check_model.py`: Script to perform model checks.

## Model Architecture Checks

The following checks are performed on the model:

1. **Total Parameter Count Test**: Ensures the model has more than 10,000 parameters.
2. **Use of Batch Normalization**: Ensures the model uses Batch Normalization layers.
3. **Use of DropOut**: Ensures the model uses Dropout layers.
4. **Use of Fully Connected Layer or GAP**: Ensures the model has Fully Connected layers or Global Average Pooling.

## Usage

1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run `train.py` to train and test the model.
4. The GitHub Actions workflow will automatically run on every push and pull request to verify the model architecture.

## Requirements

- Python 3.8
- PyTorch
- torchvision
- tqdm 