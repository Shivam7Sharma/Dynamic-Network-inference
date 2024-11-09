import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from sklearn.metrics import classification_report
from scipy.stats import entropy
import time  # Ensure this import is at the top of your script


num_classes = 10
num_layers = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
dataset = CIFAR10(root="./data", download=True, transform=ToTensor())
test_dataset = CIFAR10(root="./data", train=False, transform=ToTensor())

batch_size = 128
val_size = 5000
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size * 2, num_workers=4)
val_loader = DataLoader(val_ds, batch_size * 2, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size * 2, num_workers=4)


class Branch(nn.Module):
    def __init__(self, in_channels, in_features):
        super(Branch, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=16, kernel_size=3, stride=2
        )
        self.bn = nn.BatchNorm2d(num_features=16)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=in_features, out_features=num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.bn(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.in_channels = [32, 32, 64, 64, 128]
        self.in_features = [3600, 784, 784, 144, 144]
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, padding="same"
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding="same"
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.dropout1 = nn.Dropout(p=0.2)

        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding="same"
        )
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding="same"
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.bn4 = nn.BatchNorm2d(num_features=64)
        self.dropout2 = nn.Dropout(p=0.3)

        self.conv5 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding="same"
        )
        self.conv6 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding="same"
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.bn5 = nn.BatchNorm2d(num_features=128)
        self.bn6 = nn.BatchNorm2d(num_features=128)
        self.dropout3 = nn.Dropout(p=0.4)

        self.branch1 = Branch(
            in_channels=self.in_channels[0], in_features=self.in_features[0]
        )
        self.branch2 = Branch(
            in_channels=self.in_channels[1], in_features=self.in_features[1]
        )
        self.branch3 = Branch(
            in_channels=self.in_channels[2], in_features=self.in_features[2]
        )
        self.branch4 = Branch(
            in_channels=self.in_channels[3], in_features=self.in_features[3]
        )
        self.branch5 = Branch(
            in_channels=self.in_channels[4], in_features=self.in_features[4]
        )

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=2048, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=128)
        self.bn7 = nn.BatchNorm1d(num_features=128)
        self.dropout4 = nn.Dropout(p=0.5)
        self.fc5 = nn.Linear(in_features=128, out_features=num_classes)

        self.num_layers = num_layers

    def forward(self, tensor_after_previous_layer, exit_layer_idx=num_layers):
        if exit_layer_idx == 0:
            x = self.conv1(tensor_after_previous_layer)
            x = F.relu(x)
            tensor_after_layer = self.bn1(x)
            predicted_scores_from_layer = self.branch1(tensor_after_layer)

        elif exit_layer_idx == 1:
            x = self.conv2(tensor_after_previous_layer)
            x = F.relu(x)
            x = self.bn2(x)
            x = self.pool1(x)
            tensor_after_layer = self.dropout1(x)
            predicted_scores_from_layer = self.branch2(tensor_after_layer)

        elif exit_layer_idx == 2:
            x = self.conv3(tensor_after_previous_layer)
            x = F.relu(x)
            tensor_after_layer = self.bn3(x)
            predicted_scores_from_layer = self.branch3(tensor_after_layer)

        elif exit_layer_idx == 3:
            x = self.conv4(tensor_after_previous_layer)
            x = F.relu(x)
            x = self.bn4(x)
            x = self.pool2(x)
            tensor_after_layer = self.dropout2(x)
            predicted_scores_from_layer = self.branch4(tensor_after_layer)

        elif exit_layer_idx == 4:
            x = self.conv5(tensor_after_previous_layer)
            x = F.relu(x)
            tensor_after_layer = self.bn5(x)
            predicted_scores_from_layer = self.branch5(tensor_after_layer)

        elif exit_layer_idx == 5:
            x = self.conv6(tensor_after_previous_layer)
            x = F.relu(x)
            x = self.bn6(x)
            x = self.pool3(x)
            x = self.dropout3(x)

            x = self.flatten(x)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            x = F.relu(x)
            x = self.fc3(x)
            x = F.relu(x)
            x = self.fc4(x)
            x = F.relu(x)
            x = self.bn7(x)
            tensor_after_layer = self.dropout4(x)
            predicted_scores_from_layer = self.fc5(tensor_after_layer)

        else:
            ValueError(
                f"exit_layer_idx {exit_layer_idx} should be int within 0 to 5")

        return tensor_after_layer, predicted_scores_from_layer


model = Baseline().to(device)
model.load_state_dict(torch.load(
    "cifar10_branchyNet_m.h5", map_location="cpu"))
model.eval()


def cutoff_exit_performance_check_part1(model, dataloader, cutoff, device):
    model.eval()
    per_layer_accuracy = []
    per_layer_avg_time = []

    for exit_layer_idx in range(num_layers + 1):
        total_correct = 0
        total_samples = 0
        total_time = 0

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                current_input = images  # Starting input for the first layer

                for layer_idx in range(exit_layer_idx + 1):
                    start_time = time.time()
                    current_input, predicted_scores_from_layer = model(
                        current_input, layer_idx)
                    total_time += time.time() - start_time

                    if layer_idx == exit_layer_idx:
                        # Calculate probabilities and entropy from predicted scores
                        probabilities = F.softmax(
                            predicted_scores_from_layer, dim=1)
                        entropy_values = entropy(
                            probabilities.cpu().numpy(), axis=1)

                        # Create exit mask
                        entropy_tensor = torch.from_numpy(
                            entropy_values).to(device)
                        exit_mask = (entropy_tensor <= cutoff)

                        # Filter out exiting samples
                        exiting_output = predicted_scores_from_layer[exit_mask]
                        exiting_labels = labels[exit_mask]

                        # Calculate accuracy for exiting samples
                        _, predicted = torch.max(exiting_output, 1)
                        total_correct += (predicted ==
                                          exiting_labels).sum().item()
                        total_samples += exiting_labels.size(0)

                        # Update images and labels for the next layer
                        labels = labels[~exit_mask]

                        if not exit_mask.any():  # If no samples left, break
                            break

        layer_accuracy = total_correct / total_samples if total_samples > 0 else 0
        per_layer_accuracy.append(layer_accuracy)
        avg_time_per_sample = total_time / total_samples if total_samples > 0 else 0
        per_layer_avg_time.append(avg_time_per_sample)

    return per_layer_accuracy, per_layer_avg_time


def cutoff_exit_performance_check(model, dataloader, cutoff, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    total_time = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            for exit_layer_idx in range(num_layers + 1):
                start_time = time.time()
                tensor_after_layer, predicted_scores_from_layer = model(
                    images, exit_layer_idx)

                # Calculate probabilities and entropy from predicted scores
                probabilities = F.softmax(predicted_scores_from_layer, dim=1)
                entropy_values = entropy(probabilities.cpu().numpy(), axis=1)

                # Create exit mask
                entropy_tensor = torch.from_numpy(entropy_values).to(device)
                exit_mask = (entropy_tensor <= cutoff)

                # Filter out exiting samples
                exiting_output = predicted_scores_from_layer[exit_mask]
                exiting_labels = labels[exit_mask]

                # Calculate accuracy for exiting samples
                _, predicted = torch.max(exiting_output, 1)
                total_correct += (predicted == exiting_labels).sum().item()
                total_samples += exiting_labels.size(0)

                # Update images and labels for the next layer
                images = tensor_after_layer[~exit_mask]
                labels = labels[~exit_mask]

                total_time += time.time() - start_time

                if images.size(0) == 0:
                    break

    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
    return overall_accuracy, total_time


def estimate_thresholds(model, dataloader, desired_accuracy, device):
    model.eval()
    thresholds = []
    inference_times = []
    for exit_layer_idx in range(num_layers+1):
        entropy_correct_pairs = []

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                tensor_after_layer = images
                start_time = time.time()  # Start timing before the loop over layers

                for layer_idx in range(exit_layer_idx + 1):
                    tensor_after_layer, predicted_scores_from_layer = model(
                        tensor_after_layer, layer_idx)

                    if layer_idx == exit_layer_idx:
                        probabilities = F.softmax(
                            predicted_scores_from_layer, dim=1)
                        entropy_values = entropy(
                            probabilities.cpu().numpy(), axis=1)
                        _, predicted = torch.max(
                            predicted_scores_from_layer, 1)
                        correct = (predicted == labels).cpu().numpy()

                        for ent, corr in zip(entropy_values, correct):
                            entropy_correct_pairs.append((ent, corr))
                # End timing after processing the layer
                total_inference_time = time.time() - start_time

        # Sort and find threshold
        entropy_correct_pairs.sort(key=lambda x: x[0])
        accumulated_correct = 0
        total_samples = len(entropy_correct_pairs)
        for idx, (ent, corr) in enumerate(entropy_correct_pairs):
            accumulated_correct += corr
            if accumulated_correct / total_samples >= desired_accuracy:
                thresholds.append(ent)
                break
        else:
            # Desired accuracy not achievable, set to max entropy
            thresholds.append(0)
        # Append average inference time per sample for this layer
        avg_inference_time = total_inference_time / total_samples
        inference_times.append(avg_inference_time)

    return thresholds, inference_times


def test_with_thresholds(model, dataloader, thresholds, device):
    model.eval()
    total_time = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            start_time = time.time()  # Start timing

            for exit_layer_idx in range(num_layers + 1):
                tensor_after_layer, predicted_scores_from_layer = model(
                    images, exit_layer_idx)
                probabilities = F.softmax(predicted_scores_from_layer, dim=1)
                entropy_values = entropy(probabilities.cpu().numpy(), axis=1)

                entropy_tensor = torch.from_numpy(entropy_values).to(device)
                exit_mask = (entropy_tensor <= thresholds[exit_layer_idx])

                if exit_mask.any():
                    exiting_output = predicted_scores_from_layer[exit_mask]
                    exiting_labels = labels[exit_mask]

                    _, predicted = torch.max(exiting_output, 1)
                    total_correct += (predicted == exiting_labels).sum().item()
                    total_samples += exiting_labels.size(0)

                images = tensor_after_layer[~exit_mask]
                labels = labels[~exit_mask]

                if images.size(0) == 0:
                    break

            total_time += time.time() - start_time

    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
    return overall_accuracy, total_time


# TODO: 1(a) For a fixed value of cutoff, show performance for all layers.
# For CIFAR-10 with 10 classes
probabilities_uniform = np.full(10, 1/10)
max_entropy = entropy(probabilities_uniform)

# The minimum entropy is 0 for a deterministic prediction
min_entropy = 0


# TODO: 1(b) Plot overall accuracy vs cutoff, total time vs cutoff
# and total time vs overall accuracy.
cutoff_values = np.linspace(min_entropy, max_entropy, 100)
accuracy_values = []
total_time_values = []

cutoff = 0.6
per_layer_accuracy, per_layer_avg_time = cutoff_exit_performance_check_part1(
    model, test_loader, cutoff, device)
print(
    f"Accuracy per layer: {per_layer_accuracy}, per layer inference Time: {per_layer_avg_time}")

for cutoff in cutoff_values:
    accuracy, total_time = cutoff_exit_performance_check(
        model, test_loader, cutoff, device)
    accuracy_values.append(accuracy)
    total_time_values.append(total_time)

# Plotting the graphs
# Plotting the graphs
plt.figure(figsize=(12, 8))

# Overall Accuracy vs Cutoff
plt.subplot(2, 2, 1)
plt.plot(cutoff_values, accuracy_values, marker='o')
plt.xlabel('Cutoff')
plt.ylabel('Overall Accuracy')
plt.title('Overall Accuracy vs Cutoff')

# Total Time vs Cutoff
plt.subplot(2, 2, 2)
plt.plot(cutoff_values, total_time_values, marker='o')
plt.xlabel('Cutoff')
plt.ylabel('Total Time')
plt.title('Total Time vs Cutoff')

# Total Time vs Overall Accuracy
plt.subplot(2, 2, 3)
plt.plot(accuracy_values, total_time_values, marker='o')
plt.xlabel('Overall Accuracy')
plt.ylabel('Total Time')
plt.title('Total Time vs Overall Accuracy')

plt.tight_layout()
plt.show()
desired_accuracies = [0.8, 0.85, 0.9, 0.95]
inference_times = []
accuracies = []

for acc in desired_accuracies:
    thresholds, layer_inference_times = estimate_thresholds(
        model, val_loader, acc, device)
    print(f"Desired Accuracy: {acc}, Estimated Thresholds: {thresholds}")
    print(
        f"Layer-wise Inference Times on Validation Data: {layer_inference_times}")

    thresholds_training, layer_inference_times_training = estimate_thresholds(
        model, train_loader, acc, device)
    print(
        f"Layer-wise Inference Times on Training Data: {layer_inference_times_training}")

    accuracy, total_time = test_with_thresholds(
        model, test_loader, thresholds_training, device)
    accuracies.append(accuracy)
    inference_times.append(total_time)

    print(
        f"Accuracy Achieved on Test Data: {accuracy}, Total Inference Time: {total_time}")

# Plotting
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(desired_accuracies, inference_times,
         marker='o', label='Total Inference Time')
plt.xlabel('Desired Accuracy')
plt.ylabel('Inference Time')
plt.title('Inference Time vs Desired Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(desired_accuracies, accuracies, marker='o', label='Accuracy')
plt.xlabel('Desired Accuracy')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Desired Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# TODO: 2(a) On validation data, estimate threshold for each layer based on
# desired minimum accuracy. Use said list of thresholds on test data.


# TODO: 2(c) Vary the desired minimum accuracy and generate lists of
# thresholds. For the list of list of thresholds, plot total time
# vs overall accuracy.
