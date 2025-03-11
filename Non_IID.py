# import numpy as np
# import torch
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader, Subset

# def create_noniid_fmnist(alpha=0.5, num_classes=10, num_samples=60000):
#     # Load Fashion-MNIST dataset
#     transform = transforms.Compose([transforms.ToTensor()])
#     dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

#     labels = np.array(dataset.targets)

#     # Apply Dirichlet distribution for non-IID split
#     class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
#     class_sizes = [len(idx) for idx in class_indices]
    
#     # Dirichlet distribution to control imbalance
#     proportions = np.random.dirichlet(alpha * np.ones(num_classes))

#     print("Class proportions:", proportions)

#     selected_indices = []
#     for i, p in enumerate(proportions):
#         size = int(p * num_samples)  # Control how much to sample from each class
#         chosen_indices = np.random.choice(class_indices[i], size=size, replace=False)
#         selected_indices.extend(chosen_indices)

#     # Create a subset with non-IID distribution
#     noniid_dataset = Subset(dataset, selected_indices)
    
#     return noniid_dataset

# # Create a non-IID Fashion-MNIST dataset
# noniid_data = create_noniid_fmnist(alpha=0.3)

# # DataLoader for training
# train_loader = DataLoader(noniid_data, batch_size=64, shuffle=True)

# print("Non-IID dataset size:", len(noniid_data))



import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

def create_noniid_fmnist(alpha=10000.0, num_classes=10, num_samples=60000):
    # Load Fashion-MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

    labels = np.array(dataset.targets)

    # Apply Dirichlet distribution for non-IID split
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    class_sizes = [len(idx) for idx in class_indices]
    
    # Dirichlet distribution to control imbalance
    proportions = np.random.dirichlet(alpha * np.ones(num_classes))
    print("Class proportions:", proportions)

    selected_indices = []
    for i, p in enumerate(proportions):
        # Ensure the sample size does not exceed the available samples
        size = min(int(p * num_samples), class_sizes[i])
        
        # Avoid empty selection
        if size > 0:
            chosen_indices = np.random.choice(class_indices[i], size=size, replace=False)
            selected_indices.extend(chosen_indices)

    # Create a subset with non-IID distribution
    noniid_dataset = Subset(dataset, selected_indices)
    
    return noniid_dataset

# Create a non-IID Fashion-MNIST dataset
# noniid_data = create_noniid_fmnist(alpha=10000.0)

# DataLoader for training
# train_loader = DataLoader(noniid_data, batch_size=64, shuffle=True)

# print("Non-IID dataset size:", len(noniid_data))
