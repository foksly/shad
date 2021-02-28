import argparse
from dotmap import DotMap

import wandb
import numpy as np
import torch
from torch import nn

import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import time
import sys
sys.path.append('/home/foksly/Documents')

from shad.nn.datasets import OmniglotData, MnistData
from shad.nn.datasets.unpackers import AutoencoderUnpacker
from shad.nn.logger import WBLogger
from shad.nn.models.autoencoder import Conv2dAutoEncoder, Conv2dDenoisingAutoEncoder, Conv2dSparseKLAutoencoder, \
    LatentFeaturesDenseClassifier, Noise
from shad.nn.models.vision import Conv2dClassifier
from shad.nn.trainer import DefaultTrainer
from shad.nn.utils import load_config
from shad.nn.metrics import FID, Accuracy

from utils import SparseL1Loss, SparseKLLoss, SparseTrainer

DATA_PATH = '/home/foksly/Documents/shad/spring-2021/DeepGenerativeModels/homework/1-AE/'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-autoencoder', action='store_true')
    parser.add_argument('--train-classifier', nargs='+', default=[],
                        help='Train classifier on "omniglot", "mnist" or "latent_features"')
    parser.add_argument('--compute-fid', action='store_true')
    parser.add_argument('--autoencoder-checkpoint-path')
    parser.add_argument('--mnist-classifier-checkpoint-path')
    parser.add_argument('--config')

    return parser.parse_args()

def do_nothing():
    print('Nothing was requested', end='', flush=True)
    for _ in range(3):
        time.sleep(1)
        print('.', end='', flush=True)
    print('\nNothing was successfully done!')


@torch.no_grad()
def log_autoencoder_reconstructions(autoencoder, images, logger, device, noise=None):
    images = images.to(device)
    reconstructed_images = autoencoder(images)

    images_log = []
    for i in range(images.shape[0]):
        cat_images = [images[i]]
        if noise is not None:
            noisy_image = Noise(noise)(images[i])
            cat_images.append(noisy_image)
        cat_images.append(reconstructed_images[i])
        image = torch.cat(list(map(lambda x: x.unsqueeze(0).cpu(), cat_images)), dim=0)
        images_log.append(image)

    logger.log_image(images_log, label=f'autoencoder reconstructions')


def load_from_checkpoint(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def train_autoencoder(data, config, logger, checkpoint_path=None):
    # prepare stuff for training
    criterion = nn.MSELoss()
    Trainer = DefaultTrainer

    if config.desc.startswith('vanilla'):
        autoencoder = Conv2dAutoEncoder(**config.hp)
    elif config.desc.startswith('denoising'):
        autoencoder = Conv2dDenoisingAutoEncoder(**config.hp)
    elif config.desc.startswith('sparse-l1'):
        autoencoder = Conv2dAutoEncoder(**config.hp)
        criterion = SparseL1Loss(**config.loss)
        Trainer = SparseTrainer
    elif config.desc.startswith('sparse-kl'):
        autoencoder = Conv2dSparseKLAutoencoder(**config.hp)
        criterion = SparseKLLoss(**config.loss)
        Trainer = SparseTrainer

    optimizer = torch.optim.Adam
    device = torch.device(config.device)

    # train
    trainer = Trainer(autoencoder, optimizer, criterion, data, logger,
                      device=device, lr=config.optimizer.lr, unpacker=AutoencoderUnpacker,
                      valid=True, desc='autoencoder', checkpoint_path=checkpoint_path)
    autoencoder = trainer.train(config.epochs)

    # log images
    images, _ = next(iter(data.testloader))
    log_autoencoder_reconstructions(autoencoder, images[2:18], logger, device=device, noise=config.hp.input_noise or None)
    return autoencoder


def train_classifier(data, config, logger, checkpoint_path=None):
    # prepare stuff for training
    classifier = Conv2dClassifier(**config.hp)
    optimizer = torch.optim.Adam
    criterion = nn.CrossEntropyLoss()
    device = torch.device(config.device)
    metrics = {'accuracy': Accuracy()}

    # train
    trainer = DefaultTrainer(classifier, optimizer, criterion, data, logger,
                             device=device, lr=config.optimizer.lr, valid=True,
                             metrics=metrics, desc=config.desc, checkpoint_path=checkpoint_path)
    classifier = trainer.train(config.epochs)

    return classifier


def get_latent_features(dataloader, autoencoder, device):
    autoencoder.to(device)
    features, targets = [], []
    for images, labels in dataloader:
        images = images.to(device)
        with torch.no_grad():
            features.append(autoencoder.get_latent_features(images, flatten=True).cpu().numpy())
            targets.append(labels.numpy())

    features = np.vstack(features)
    targets = np.concatenate(targets)
    return features, targets


def train_fit_predict_on_latent_features(data, autoencoder, device):
    # get features
    train_features, train_labels = get_latent_features(data.trainloader, autoencoder, device)
    test_features, test_labels = get_latent_features(data.testloader, autoencoder, device)

    # train catboost classifier
    catboost_clf = CatBoostClassifier(iterations=100, task_type='GPU', max_depth=3)
    catboost_clf.fit(train_features, train_labels)

    catboost_pred = catboost_clf.predict(test_features)
    catboost_accuracy = accuracy_score(test_labels, catboost_pred)

    # train random forest
    rf_clf = RandomForestClassifier(n_estimators=200, max_depth=30, verbose=4, n_jobs=-1)
    rf_clf.fit(train_features, train_labels)

    rf_pred = rf_clf.predict(test_features)
    rf_accuracy = accuracy_score(test_labels, rf_pred)

    table = wandb.Table(data=[[round(catboost_accuracy, 3), round(rf_accuracy, 3)]], columns=["Catboost", "Random Forest"])
    wandb.log({"Forests Ensembles": table})
    wandb.log({"Catboost accuracy": catboost_accuracy, "Random Forest accuracy": rf_accuracy})

def train_nn_on_latent_features(data, autoencoder, logger, device):
    # obtain autoencoder latent dim size
    autoencoder.to(device)
    dummy_image = torch.rand(1, 1, 64, 64, device=device)
    in_features = np.prod(autoencoder.encoder(dummy_image).shape)

    # prepare stuff for training
    classifier = LatentFeaturesDenseClassifier(autoencoder, in_features, data.n_classes)
    optimizer = torch.optim.Adam
    criterion = nn.CrossEntropyLoss()
    metrics = {'accuracy': Accuracy()}

    # train
    trainer = DefaultTrainer(classifier, optimizer, criterion, data, logger,
                             device=device, lr=5e-4, valid=True,
                             metrics=metrics, desc='nn_classifier')
    classifier = trainer.train(30)

    return classifier


def main():
    args = parse_args()
    if not (args.train_autoencoder or args.train_classifier or args.compute_fid):
        return do_nothing()

    # load config
    config = load_config(args.config)

    # load data
    data = {}
    if 'mnist' not in args.train_classifier:
        data['omniglot'] = OmniglotData(download=False, download_path=f'{DATA_PATH}/Omniglot')
        data['omniglot'].build_dataloaders(batch_size=config.data.omniglot.batch_size)

    if 'mnist' in args.train_classifier:
        data['mnist'] = MnistData(download=False, download_path=f'{DATA_PATH}/MNIST', image_size=64)
        data['mnist'].build_dataloaders(batch_size=config.data.mnist.batch_size)

    # set logger
    logger = WBLogger(project=config.logger.project, config=config, run_name=config.logger.run_name)

    # train autoencoder
    if args.train_autoencoder and (args.autoencoder_checkpoint_path is None):
        autoencoder_checkpoint_path = f'checkpoints/{config.train.autoencoder.desc}_autoencoder.pt'
        autoencoder = train_autoencoder(data['omniglot'], config.train.autoencoder, logger, autoencoder_checkpoint_path)

    # train MNIST classifier
    if 'mnist' in args.train_classifier and (args.mnist_classifier_checkpoint_path is None):
        mnist_checkpoint_path = f'checkpoints/{config.train.mnist_classifier.desc}.pt'
        mnist_classifier = train_classifier(data['mnist'], config.train.mnist_classifier, logger, mnist_checkpoint_path)

    # train OMNIGLOT classifier
    if 'omniglot' in args.train_classifier:
        omniglot_checkpoint_path = f'checkpoints/{config.train.omniglot_classifier.desc}.pt'
        omniglot_classifier = train_classifier(data['omniglot'], config.train.omniglot_classifier, logger, omniglot_checkpoint_path)

    # load checkpoints
    if args.autoencoder_checkpoint_path is not None:
        if config.train.autoencoder.desc.startswith('vanilla') or config.train.autoencoder.desc.startswith('sparse-l1'):
            autoencoder = Conv2dAutoEncoder()
        elif config.train.autoencoder.desc.startswith('denoising'):
            autoencoder = Conv2dDenoisingAutoEncoder(**config.train.autoencoder.hp)
        elif config.train.autoencoder.desc.startswith('sparse-kl'):
            autoencoder = Conv2dSparseKLAutoencoder(**config.train.autoencoder.hp)

        autoencoder = load_from_checkpoint(autoencoder, args.autoencoder_checkpoint_path)

    if args.mnist_classifier_checkpoint_path is not None:
        mnist_classifier = Conv2dClassifier(**config.train.mnist_classifier.hp)
        mnist_classifier = load_from_checkpoint(mnist_classifier, args.mnist_classifier_checkpoint_path)

    # compute FID
    if args.compute_fid:
        scorer = FID()
        fid_score = scorer(data['omniglot'].testloader, autoencoder, mnist_classifier)

        table = wandb.Table(data=[[round(fid_score, 3)]], columns=["FID"])
        wandb.log({"FID score": table})
        wandb.log({'FID': fid_score})

    # train classifier on latent features
    if 'latent_features' in args.train_classifier:
        device = torch.device('cuda:0')
        train_fit_predict_on_latent_features(data['omniglot'], autoencoder, device)
        train_nn_on_latent_features(data['omniglot'], autoencoder, logger, device)


if __name__ == "__main__":
    main()
