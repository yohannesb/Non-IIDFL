# import pickle
# import numpy as np
# import torch
# import sys
# from torch.utils.data import DataLoader, Dataset, Subset
# from torchvision import datasets, transforms
# from loguru import logger
# from .dataset import Dataset

# # Configure Logger
# logger.remove()
# logger.add("label_flipping_log.txt", format="{time} | {level} | {message}", level="DEBUG")

# # Add handler for terminal output (command prompt)
# logger.add(sys.stdout, format="{time} | {level} | {message}", level="DEBUG")

# class CIFAR10Dataset(Dataset):

#     def __init__(self, args):
#         super(CIFAR10Dataset, self).__init__(args)

#     def load_train_dataset(self):
#         self.get_args().get_logger().debug("Loading CIFAR10 train data")

#         normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         transform = transforms.Compose([
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomCrop(32, 4),
#             transforms.ToTensor(),
#             normalize
#         ])
#         train_dataset = datasets.CIFAR10(root=self.get_args().get_data_path(), train=True, download=True, transform=transform)

#         # Apply Label Flipping
#         #self.flip_labels(train_dataset)

#         # Apply Pathological Non-IID Distribution
#         train_dataset = self.create_noniid_partitions(train_dataset, num_clients=10, num_classes_per_client=2)

#         train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
#         train_data = self.get_tuple_from_data_loader(train_loader)

#         self.get_args().get_logger().debug("Finished loading CIFAR10 train data")
#         return train_data

#     def load_test_dataset(self):
#         self.get_args().get_logger().debug("Loading CIFAR10 test data")

#         normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#             normalize
#         ])
#         test_dataset = datasets.CIFAR10(root=self.get_args().get_data_path(), train=False, download=True, transform=transform)

#         # Apply Label Flipping (if required for test set)
#         #self.flip_labels(test_dataset)

#         test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
#         test_data = self.get_tuple_from_data_loader(test_loader)

#         self.get_args().get_logger().debug("Finished loading CIFAR10 test data")
#         return test_data

#     def flip_labels(self, dataset, flip_percentage=0.2, source_class=5, target_class=3):
#         targets = np.array(dataset.targets)

#         # Find samples of the source class
#         class_indices = np.where(targets == source_class)[0]
#         num_to_flip = int(len(class_indices) * flip_percentage)

#         # Randomly select samples to flip
#         flip_indices = np.random.choice(class_indices, num_to_flip, replace=False)

#         # Perform label flipping
#         targets[flip_indices] = target_class
#         dataset.targets = targets.tolist()

#         logger.warning(f"⚠️ Flipped {num_to_flip} samples from Class {source_class} → Class {target_class}")

#     def create_noniid_partitions(self, dataset, num_clients=10, num_classes_per_client=2):
#         """
#         Creates pathological non-IID partitions by limiting each client to a subset of classes.
#         """
#         targets = np.array(dataset.targets)
#         class_indices = [np.where(targets == i)[0] for i in range(10)]

#         client_datasets = []

#         for client_id in range(num_clients):
#             chosen_classes = np.random.choice(range(10), num_classes_per_client, replace=False)
#             indices = np.concatenate([class_indices[c] for c in chosen_classes])

#             client_subset = Subset(dataset, indices)
#             client_datasets.append(client_subset)

#             logger.info(f"Client {client_id}: Classes {chosen_classes} | Samples {len(indices)}")

#         # For simplicity, returning a merged dataset (could return individual client datasets for true FL setup)
#         return torch.utils.data.ConcatDataset(client_datasets)

# # Save the processed dataset
# def save_dataset(dataset, file_path):
#     with open(file_path, 'wb') as f:
#         pickle.dump(dataset, f)
#     logger.success(f"🎉 Saved dataset to {file_path}")

# # Initialize and save CIFAR-10 dataset
# def main(args):
#     cifar_dataset = CIFAR10Dataset(args)

#     # Load and save the train dataset
#     train_data = cifar_dataset.load_train_dataset()
#     save_dataset(train_data, "train_data_loader.pickle")

#     # Load and save the test dataset
#     test_data = cifar_dataset.load_test_dataset()
#     save_dataset(test_data, "test_data_loader.pickle")

#     logger.success("✅ Dataset processing and saving complete.")



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
        # self.flip_labels(train_dataset)

        # Apply Non-IID Distribution using Dirichlet
        train_dataset = self.create_noniid_partitions(train_dataset, num_clients=10, alpha=0.5)

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
        # self.flip_labels(test_dataset)

        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
        test_data = self.get_tuple_from_data_loader(test_loader)

        self.get_args().get_logger().debug("Finished loading CIFAR10 test data")
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
        """
        Creates non-IID partitions using Dirichlet distribution to simulate realistic data heterogeneity.

        Parameters:
        - dataset: The CIFAR-10 dataset.
        - num_clients: Number of clients to simulate.
        - alpha: Dirichlet distribution parameter (smaller alpha = more imbalance).

        Returns:
        - Merged dataset combining all client subsets (or separate client datasets if needed).
        """
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
