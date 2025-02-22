import torchvision
import torchvision.transforms as transforms
import torch

def load_data(batch_size=256):
    transform = transforms.Compose([
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    eval_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    eval_loader = torch.utils.data.DataLoader(eval_set, batch_size=batch_size)

    return train_loader, eval_loader
