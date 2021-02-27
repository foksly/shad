from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

N_CLASSES = 10

class MnistData:
    def __init__(self, download=False, download_path='./MNIST', image_size=32):
        self.n_classes = N_CLASSES

        self.transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        self.train_dataset = MNIST(download_path, train=True, transform=self.transforms, download=download)
        self.test_dataset = MNIST(download_path, train=False, transform=self.transforms, download=download)

    def build_dataloaders(self, batch_size=64, drop_last=True):
        self.trainloader = DataLoader(self.train_dataset, batch_size=batch_size, drop_last=drop_last, shuffle=True)
        self.testloader = self.validloader = DataLoader(self.test_dataset, batch_size=batch_size, drop_last=drop_last, shuffle=True)
