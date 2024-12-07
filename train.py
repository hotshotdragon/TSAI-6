import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.mnist_net import Net
from utils.training import train, test
import logging

def main():
    # Set up logging
    logging.basicConfig(filename='train_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    torch.manual_seed(1)
    batch_size = 28 

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.RandomRotation((-15, 15)),
                        transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, epochs=20, steps_per_epoch=len(train_loader))

    for epoch in range(1, 21):
        logging.info(f"Epoch {epoch}")
        print(f"Epoch {epoch}")
        train(model, device, train_loader, optimizer, epoch)
        accuracy = test(model, device, test_loader)
        
        # Log accuracy
        logging.info(f"Test Accuracy: {accuracy:.2f}%")
        
        # Stop training if accuracy exceeds 99.4%
        if accuracy > 99.4:
            logging.info(f"Stopping training as test accuracy reached {accuracy:.2f}%")
            print(f"Stopping training as test accuracy reached {accuracy:.2f}%")
            break

if __name__ == "__main__":
    main() 