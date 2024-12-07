# MNIST Model Training

![Build Status](https://img.shields.io/github/actions/workflow/status/yourusername/yourrepo/model_checks.yml?branch=main)
![Python Version](https://img.shields.io/badge/python-3.8-blue)

This project contains a PyTorch implementation of a convolutional neural network (CNN) for classifying the MNIST dataset. The model is defined in a modular format, and GitHub Actions are used to ensure the model architecture meets specific requirements.

## Project Structure

- `models/mnist_net.py`: Contains the model definition.
- `utils/training.py`: Contains utility functions for training and testing the model.
- `train.py`: Main script to train and test the model.
- `.github/workflows/model_checks.yml`: GitHub Actions workflow to check model architecture.
- `tests/test_model.py`: Script to perform model checks.
- `train_log.txt`: Log file generated during training.

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

## Viewing Training Logs

During training, logs are generated and saved to `train_log.txt`. These logs include information about each training epoch and test accuracy. You can view the logs by opening the `train_log.txt` file.

### Sample Log Output
Epoch 1
loss=0.3253727853298187 batch_id=2142: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2143/2143 [00:14<00:00, 149.89it/s]

Test set: Average loss: 0.0693, Accuracy: 9794/10000 (97.94%)

Epoch 2
loss=0.1257178634405136 batch_id=2142: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2143/2143 [00:13<00:00, 159.39it/s]

Test set: Average loss: 0.0450, Accuracy: 9871/10000 (98.71%)

Epoch 3
loss=0.0744318887591362 batch_id=2142: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2143/2143 [00:13<00:00, 159.99it/s]

Test set: Average loss: 0.0381, Accuracy: 9879/10000 (98.79%)

Epoch 4
loss=0.008059755899012089 batch_id=2142: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2143/2143 [00:14<00:00, 144.63it/s]

Test set: Average loss: 0.0277, Accuracy: 9920/10000 (99.20%)

Epoch 5
loss=0.36019575595855713 batch_id=2142: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2143/2143 [00:13<00:00, 160.02it/s] 

Test set: Average loss: 0.0292, Accuracy: 9911/10000 (99.11%)

Epoch 6
loss=0.008818539790809155 batch_id=2142: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2143/2143 [00:13<00:00, 161.70it/s] 

Test set: Average loss: 0.0279, Accuracy: 9911/10000 (99.11%)

Epoch 7
loss=0.024180874228477478 batch_id=2142: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2143/2143 [00:13<00:00, 163.83it/s] 

Test set: Average loss: 0.0214, Accuracy: 9928/10000 (99.28%)

Epoch 8
loss=0.16873882710933685 batch_id=2142: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2143/2143 [00:13<00:00, 156.48it/s] 

Test set: Average loss: 0.0316, Accuracy: 9894/10000 (98.94%)

Epoch 9
loss=0.12547487020492554 batch_id=2142: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2143/2143 [00:13<00:00, 160.06it/s] 

Test set: Average loss: 0.0261, Accuracy: 9920/10000 (99.20%)

Epoch 10
loss=0.03965074196457863 batch_id=2142: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2143/2143 [00:13<00:00, 162.44it/s] 

Test set: Average loss: 0.0266, Accuracy: 9923/10000 (99.23%)

Epoch 11
loss=0.01793784089386463 batch_id=2142: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2143/2143 [00:13<00:00, 164.52it/s] 

Test set: Average loss: 0.0256, Accuracy: 9929/10000 (99.29%)

Epoch 12
loss=0.09722863882780075 batch_id=2142: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2143/2143 [00:13<00:00, 162.74it/s] 

Test set: Average loss: 0.0234, Accuracy: 9934/10000 (99.34%)

Epoch 13
loss=0.032397907227277756 batch_id=2142: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2143/2143 [00:12<00:00, 166.76it/s] 

Test set: Average loss: 0.0253, Accuracy: 9931/10000 (99.31%)

Epoch 14
loss=0.008292009122669697 batch_id=2142: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2143/2143 [00:13<00:00, 162.22it/s] 

Test set: Average loss: 0.0230, Accuracy: 9926/10000 (99.26%)

Epoch 15
loss=0.08116432279348373 batch_id=2142: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2143/2143 [00:12<00:00, 165.68it/s] 

Test set: Average loss: 0.0259, Accuracy: 9915/10000 (99.15%)

Epoch 16
loss=0.027724547311663628 batch_id=2142: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2143/2143 [00:12<00:00, 164.99it/s] 

Test set: Average loss: 0.0223, Accuracy: 9931/10000 (99.31%)

Epoch 17
loss=0.023061588406562805 batch_id=2142: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2143/2143 [00:12<00:00, 165.41it/s] 

Test set: Average loss: 0.0212, Accuracy: 9936/10000 (99.36%)

Epoch 18
loss=0.0031298224348574877 batch_id=2142: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2143/2143 [00:12<00:00, 166.32it/s] 

Test set: Average loss: 0.0198, Accuracy: 9939/10000 (99.39%)

Epoch 19
loss=0.001641132403165102 batch_id=2142: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2143/2143 [00:13<00:00, 164.54it/s] 

Test set: Average loss: 0.0206, Accuracy: 9929/10000 (99.29%)

Epoch 20
loss=0.18297308683395386 batch_id=2142: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2143/2143 [00:12<00:00, 166.16it/s] 

Test set: Average loss: 0.0175, Accuracy: 9941/10000 (99.41%)

Stopping training as test accuracy reached 99.41%