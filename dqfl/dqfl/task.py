"""dqfl: A Flower / PyTorch app."""

from collections import OrderedDict
import random
import numpy as np
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from flwr_datasets.visualization import plot_label_distributions
from flwr.common import Context


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


fds = None  # Cache FederatedDataset


def load_data(partition_id: int, num_partitions: int, context : Context):
    """Load partition FashionMNIST data."""
    # Only initialize `FederatedDataset` once
    global fds
    # Set the seed for reproducibility
    seed = 42
    torch.manual_seed(seed)  # For PyTorch
    np.random.seed(seed)  # For NumPy
    random.seed(seed)  # For Python's random module

    # If you're using CUDA (GPU), also set the seed for it
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If you use multiple GPUs

    # Set the deterministic flag to ensure reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if fds is None:
        partitioner = DirichletPartitioner(num_partitions=num_partitions, partition_by="label", alpha=context.run_config["alpha"], seed=42) #small alpha -> non IID
        # Could I show the distribution in each setting? Describe and evaluate
        fds = FederatedDataset(
            dataset="zalando-datasets/fashion_mnist",
            partitioners={"train": partitioner},
            seed=42
        )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, ), (0.5, ))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)

    # Now, define your DataLoader
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True,
                             worker_init_fn=np.random.seed(seed))
    testloader = DataLoader(partition_train_test["test"], batch_size=32)



    # Plot the label distribution for train dataset
    fig, ax, df = plot_label_distributions(
        partitioner= fds.partitioners["train"],
        label_name="label",
        plot_type="bar",
        size_unit="absolute",
        partition_id_axis="x",
        legend=True,
        verbose_labels=True,
        title="Per Client Labels Distribution",
    )

    # Save the figure to a file
    fig.savefig("label_distribution.png", dpi=300, bbox_inches="tight")

    return trainloader, testloader


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["image"]
            labels = batch["label"]
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
