import torch
import torch.nn.functional as F
from torch import nn


from dropconnect import WeightDropLinear

class NN_dropconnect(nn.Module):
    def __init__(self, input_size, num_classes):
        """
        Define the layers of the neural network.

        Parameters:
            input_size: int
                The size of the input, in this case 784 (28 x 28).
            num_classes: int
                The number of classes we want to predict, in our case 10 (digits 0 to 9).
        """
        super(NN_dropconnect, self).__init__()
        self.fc1 = WeightDropLinear(input_size, 50, weight_dropout=.30)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        """
        Define the forward pass of the neural network.

        Parameters:
            x: torch.Tensor
                The input tensor.

        Returns:
            torch.Tensor
                The output tensor after passing through the network.
        """
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        """
        Define the layers of the neural network.

        Parameters:
            input_size: int
                The size of the input, in this case 784 (28 x 28).
            num_classes: int
                The number of classes we want to predict, in our case 10 (digits 0 to 9).
        """
        super(NN, self).__init__()

        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        """
        Define the forward pass of the neural network.

        Parameters:
            x: torch.Tensor
                The input tensor.

        Returns:
            torch.Tensor
                The output tensor after passing through the network.
        """
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def check_accuracy(loader, model, device):
    """
    Checks the accuracy of the model on the given dataset loader.

    Parameters:
        loader: DataLoader
            The DataLoader for the dataset to check accuracy on.
        model: nn.Module
            The neural network model.
    """
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient calculation
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            x = x.reshape(x.shape[0], -1)

            # Forward pass: compute the model output
            scores = model(x)
            _, predictions = scores.max(1)  # Get the index of the max log-probability
            num_correct += (predictions == y).sum()  # Count correct predictions
            num_samples += predictions.size(0)  # Count total samples

        # Calculate accuracy
        accuracy = float(num_correct) / float(num_samples) * 100
        print(f"Got {num_correct}/{num_samples} with accuracy {accuracy:.2f}%")

    model.train()  # Set the model back to training mode
    return num_correct, num_samples, accuracy


