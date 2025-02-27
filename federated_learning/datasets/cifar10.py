import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from loguru import logger
from .dataset import Dataset

# Configure Logger
logger.remove()
logger.add("label_flipping_log.txt", format="{time} | {level} | {message}", level="DEBUG")

class CIFAR10Dataset(Dataset):

    def __init__(self, args):
        super(CIFAR10Dataset, self).__init__(args)

    def load_train_dataset(self):
        self.get_args().get_logger().debug("Loading CIFAR10 train data")

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize
        ])
        train_dataset = datasets.CIFAR10(root=self.get_args().get_data_path(), train=True, download=True, transform=transform)

        # Apply Label Flipping
        self.flip_labels(train_dataset)

        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))

        train_data = self.get_tuple_from_data_loader(train_loader)

        self.get_args().get_logger().debug("Finished loading CIFAR10 train data")
        return train_data

    def load_test_dataset(self):
        self.get_args().get_logger().debug("Loading CIFAR10 test data")

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        test_dataset = datasets.CIFAR10(root=self.get_args().get_data_path(), train=False, download=True, transform=transform)

        # Apply Label Flipping (if required for test set)
        self.flip_labels(test_dataset)

        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

        test_data = self.get_tuple_from_data_loader(test_loader)

        self.get_args().get_logger().debug("Finished loading CIFAR10 test data")
        return test_data

    def flip_labels(self, dataset, flip_percentage=0.2, source_class=5, target_class=3):
        """
        Flips a given percentage of labels from source_class to target_class in the dataset.
        """
        targets = np.array(dataset.targets)

        # Find samples of the source class
        class_indices = np.where(targets == source_class)[0]
        num_to_flip = int(len(class_indices) * flip_percentage)

        # Randomly select samples to flip
        flip_indices = np.random.choice(class_indices, num_to_flip, replace=False)

        # Perform label flipping
        targets[flip_indices] = target_class
        dataset.targets = targets.tolist()

        # Log label flipping information
        logger.warning(f"⚠️ Flipped {num_to_flip} samples from Class {source_class} → Class {target_class}")

# Save the processed dataset
def save_dataset(dataset, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(dataset, f)
    logger.success(f"🎉 Saved dataset to {file_path}")

# Initialize and save CIFAR-10 dataset
def main(args):
    cifar_dataset = CIFAR10Dataset(args)

    # Load and save the train dataset
    train_data = cifar_dataset.load_train_dataset()
    save_dataset(train_data, "train_data_loader.pickle")

    # Load and save the test dataset
    test_data = cifar_dataset.load_test_dataset()
    save_dataset(test_data, "test_data_loader.pickle")

    logger.success("✅ Dataset processing and saving complete.")
