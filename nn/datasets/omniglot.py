# from https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/datasets/full_omniglot.py

import os

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from torch.utils.data import Dataset, ConcatDataset
from torchvision.datasets.omniglot import Omniglot
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

N_CLASSES = 1623
SEED = 92

class FullOmniglot(Dataset):
    """

    [[Source]]()

    **Description**

    This class provides an interface to the Omniglot dataset.

    The Omniglot dataset was introduced by Lake et al., 2015.
    Omniglot consists of 1623 character classes from 50 different alphabets, each containing 20 samples.
    While the original dataset is separated in background and evaluation sets,
    this class concatenates both sets and leaves to the user the choice of classes splitting
    as was done in Ravi and Larochelle, 2017.
    The background and evaluation splits are available in the `torchvision` package.

    **References**

    1. Lake et al. 2015. “Human-Level Concept Learning through Probabilistic Program Induction.” Science.
    2. Ravi and Larochelle. 2017. “Optimization as a Model for Few-Shot Learning.” ICLR.

    **Arguments**

    * **root** (str) - Path to download the data.
    * **transform** (Transform, *optional*, default=None) - Input pre-processing.
    * **target_transform** (Transform, *optional*, default=None) - Target pre-processing.
    * **download** (bool, *optional*, default=False) - Whether to download the dataset.

    **Example**
    ~~~python
    omniglot = FullOmniglot(root='./data',
                            transform=transforms.Compose([
                                transforms.Resize(28, interpolation=LANCZOS),
                                transforms.ToTensor(),
                                lambda x: 1.0 - x]),
                            download=True)
    ~~~

    """

    def __init__(self, root, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        # Set up both the background and eval dataset
        omni_background = Omniglot(self.root, background=True, download=download)
        # Eval labels also start from 0.
        # It's important to add 964 to label values in eval so they don't overwrite background dataset.
        omni_evaluation = Omniglot(self.root,
                                   background=False,
                                   download=download,
                                   target_transform=lambda x: x + len(omni_background._characters))

        self.dataset = ConcatDataset((omni_background, omni_evaluation))
        self._bookkeeping_path = os.path.join(self.root, 'omniglot-bookkeeping.pkl')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, character_class = self.dataset[item]
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            character_class = self.target_transform(character_class)

        return image, character_class

class InvertImage:
    def __call__(self, pic):
        return 1 - pic

    def __repr__(self):
        return self.__class__.__name__ + '()'


class OmniglotData:
    def __init__(self, download_path='./Omniglot', download=True, val=False, test=True):
        self.n_classes = N_CLASSES

        self.transforms = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            InvertImage(),
            transforms.Normalize([0.5], [0.5])
        ])

        self.dataset = FullOmniglot(download_path, transform=self.transforms, download=download)

    def build_dataloaders(self, batch_size=64, testloader=True):
        if not testloader:
            self.trainloader = DataLoader(self.dataset, batch_size=batch_size, drop_last=True, shuffle=True)

        labels = []
        for _, label in self.dataset:
            labels.append(label)

        indices_train, indices_test, _, _ = train_test_split(list(range(len(labels))), labels,
                                                             test_size=0.1, stratify=labels, random_state=SEED)

        train_dataset = Subset(self.dataset, indices=indices_train)
        test_dataset = Subset(self.dataset, indices=indices_test)

        self.trainloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
        self.testloader = self.validloader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True)

    def _process_image(image):
        return (image.squeeze().detach().to('cpu') + 1) / 2

    def plot_samples(self):
        trainloader = self.get_dataloader(testloader=False)
        sample_batch = next(iter(trainloader))
        plt.figure(figsize=(15, 15))
        for ind, (image, label) in enumerate(zip(*sample_batch)):
            if ind >= 25:
                break

            plt.subplot(5, 5, ind + 1)
            plt.imshow(self._process_image(image))
            plt.title(label.item())
        plt.show()
