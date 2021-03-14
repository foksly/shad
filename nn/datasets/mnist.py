from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

N_CLASSES = 10

class MnistData:
    def __init__(self, data_path='./MNIST', download=False, image_size=32, batch_size=64):
        self.n_classes = N_CLASSES

        self.transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        self.train_dataset = MNIST(data_path, train=True, transform=self.transforms, download=download)
        self.test_dataset = MNIST(data_path, train=False, transform=self.transforms, download=download)
        self.build_dataloaders(batch_size)

    def build_dataloaders(self, batch_size=64, drop_last=True):
        self.trainloader = DataLoader(self.train_dataset, batch_size=batch_size, drop_last=drop_last, shuffle=True)
        self.testloader = self.validloader = DataLoader(self.test_dataset, batch_size=batch_size, drop_last=drop_last, shuffle=True)
