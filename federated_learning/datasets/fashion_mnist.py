import pickle
import numpy as np
import torch
import sys
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from loguru import logger
from .dataset import Dataset

# Configure Logger
logger.remove()
logger.add("label_flipping_log.txt", format="{time} | {level} | {message}", level="DEBUG")

# Add handler for terminal output (command prompt)
logger.add(sys.stdout, format="{time} | {level} | {message}", level="DEBUG")

class FashionMNISTDataset(Dataset):

    def __init__(self, args):
        super(FashionMNISTDataset, self).__init__(args)

    def load_train_dataset(self):
        self.get_args().get_logger().debug("Loading Fashion MNIST train data")

        # normalize = transforms.Normalize(mean=[0.5], std=[0.5])  # Mean and std for Fashion MNIST
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ])
        train_dataset = datasets.FashionMNIST(root=self.get_args().get_data_path(), train=True, download=True, transform=transform)

        # Apply Label Flipping
        # self.flip_labels(train_dataset)

        # Apply Non-IID Distribution using Dirichlet
        train_dataset = self.create_noniid_partitions(train_dataset, num_clients=10, alpha=0.5)

        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
        train_data = self.get_tuple_from_data_loader(train_loader)

        self.get_args().get_logger().debug("Finished loading Fashion MNIST train data")
        return train_data

    def load_test_dataset(self):
        self.get_args().get_logger().debug("Loading Fashion MNIST test data")

        # normalize = transforms.Normalize(mean=[0.5], std=[0.5])  # Mean and std for Fashion MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_dataset = datasets.FashionMNIST(root=self.get_args().get_data_path(), train=False, download=True, transform=transform)

        # Apply Label Flipping (if required for test set)
        # self.flip_labels(test_dataset)

        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
        test_data = self.get_tuple_from_data_loader(test_loader)

        self.get_args().get_logger().debug("Finished loading Fashion MNIST test data")
        return test_data

    def flip_labels(self, dataset, flip_percentage=0.2, source_class=5, target_class=3):
        targets = np.array(dataset.targets)

        # Find samples of the source class
        class_indices = np.where(targets == source_class)[0]
        num_to_flip = int(len(class_indices) * flip_percentage)

        # Randomly select samples to flip
        flip_indices = np.random.choice(class_indices, num_to_flip, replace=False)

        # Perform label flipping
        targets[flip_indices] = target_class
        dataset.targets = targets.tolist()

        logger.warning(f"⚠️ Flipped {num_to_flip} samples from Class {source_class} → Class {target_class}")

    def create_noniid_partitions(self, dataset, num_clients=10, alpha=0.5):
        targets = np.array(dataset.targets)
        num_classes = 10

        # Get indices for each class
        class_indices = [np.where(targets == i)[0] for i in range(num_classes)]

        # Dirichlet distribution: Allocate samples to clients
        class_proportions = np.random.dirichlet([alpha] * num_clients, num_classes)

        client_datasets = []

        for client_id in range(num_clients):
            client_indices = []

            for class_id in range(num_classes):
                # Determine how many samples to take from each class
                class_size = len(class_indices[class_id])
                num_samples = int(class_proportions[class_id, client_id] * class_size)

                # Ensure we don't exceed available samples
                num_samples = min(num_samples, len(class_indices[class_id]))

                # Randomly select samples without replacement
                chosen_indices = np.random.choice(class_indices[class_id], num_samples, replace=False)

                # Remove selected indices to avoid duplicate selection
                class_indices[class_id] = np.setdiff1d(class_indices[class_id], chosen_indices)

                client_indices.extend(chosen_indices)

            # Create subset for the client
            client_subset = Subset(dataset, client_indices)
            client_datasets.append(client_subset)

            logger.info(f"Client {client_id}: Samples {len(client_indices)}")

        # For simplicity, returning a merged dataset (can return separate client datasets if needed)
        return torch.utils.data.ConcatDataset(client_datasets)
    
# Save the processed dataset
def save_dataset(dataset, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(dataset, f)
    logger.success(f"🎉 Saved dataset to {file_path}")

# Initialize and save Fashion MNIST dataset
def main(args):
    fashion_mnist_dataset = FashionMNISTDataset(args)

    # Load and save the train dataset
    train_data = fashion_mnist_dataset.load_train_dataset()
    save_dataset(train_data, "train_data_loader.pickle")

    # Load and save the test dataset
    test_data = fashion_mnist_dataset.load_test_dataset()
    save_dataset(test_data, "test_data_loader.pickle")

    logger.success("✅ Dataset processing and saving complete.")

# # Ensure this script runs as a standalone program
# if __name__ == "__main__":
#     # Placeholder to parse arguments or initialize Args as needed.
#     args = ...  # Replace with actual argument parsing if necessary.
#     main(args)
