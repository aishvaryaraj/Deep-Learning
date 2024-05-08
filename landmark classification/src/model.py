import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.3) -> None:

        super().__init__()

     
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))

        #3x224x224
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3,padding=1)#16x224,224
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)#16x112x112
        
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2) 
        
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(dropout)
        
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(dropout)
        self.relu4 = nn.ReLU()
        
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.batchnorm5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(2, 2)
        self.relu5 = nn.ReLU()
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 7 * 7, 1000)
        self.dp1 = nn.Dropout(dropout)
        self.rl1 = nn.ReLU()
        self.fc2 = nn.Linear(1000, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #  process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        
        x = self.pool1(self.relu1(self.batchnorm1(self.conv1(x))))
        x = self.pool2(self.relu2(self.batchnorm2(self.conv2(x))))
        x = self.dropout1(self.pool3(self.relu3(self.batchnorm3(self.conv3(x)))))
        x = self.relu4(self.dropout2(self.pool4(self.batchnorm4(self.conv4(x)))))
        x = self.relu5(self.pool5(self.batchnorm5(self.conv5(x))))
        x = self.flatten(x)
        x = self.rl1(self.dp1(self.fc1(x)))
        x = self.fc2(x)
        
        return x


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
