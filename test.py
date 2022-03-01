from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.nn import Sequential, Conv2d, MaxPool2d
from torch.utils.data import DataLoader, random_split

if __name__ == "__main__":
    TRAINING_DATASET_LENGTH = 40000
    VALIDATION_DATASET_LENGTH = 10000
    BATCH_SIZE = 128

    dataset = CIFAR10('data/', transform=ToTensor())

    training_dataset, validation_dataset = random_split(dataset, [TRAINING_DATASET_LENGTH, VALIDATION_DATASET_LENGTH])
    training_dataset_loader = DataLoader(training_dataset, BATCH_SIZE)
    validation_dataset_loader = DataLoader(validation_dataset, BATCH_SIZE)

    convolution_layer = Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
    pooling_layer = MaxPool2d(2, 2)

    simple_model = Sequential(
        convolution_layer,
        pooling_layer
    )
    for images, labels in training_dataset_loader:
        print("images.shape:", images.shape)
        output = convolution_layer(images)
        output = pooling_layer(output)
        print("output.shape:", output.shape)
        break