from torch.utils.data import DataLoader, random_split
from torch.nn import Sequential, ReLU, MaxPool2d, Conv2d, Flatten, Linear
from torch.optim import Adam
from genericFunctions import GenericFunctions, DeviceDataLoader
from ImageClassificationBase import ImageClassificationBase
import torch


class ConvolutionalNetwork2(ImageClassificationBase):
    def __init__(self, input_channels=3, kernel_size=3, intermediate_channels=(32, 64, 128, 256)):
        super(ConvolutionalNetwork2, self).__init__()
        self.network = Sequential(
            Conv2d(input_channels, intermediate_channels[0], kernel_size=3, padding=1),
            ReLU(),
            Conv2d(intermediate_channels[0], intermediate_channels[1], kernel_size=kernel_size, padding=1),
            ReLU(),
            MaxPool2d(2, 2),
            Conv2d(intermediate_channels[1], intermediate_channels[2], kernel_size=kernel_size, padding=1),
            ReLU(),
            Conv2d(intermediate_channels[2], intermediate_channels[2], kernel_size=kernel_size, padding=1),
            ReLU(),
            MaxPool2d(2, 2),
            Conv2d(intermediate_channels[2], intermediate_channels[3], kernel_size=kernel_size, padding=1),
            ReLU(),
            Conv2d(intermediate_channels[3], intermediate_channels[3], kernel_size=kernel_size, padding=1),
            ReLU(),
            MaxPool2d(2, 2),
            Flatten(),
            Linear(256 * 4 * 4, 1024),
            ReLU(),
            Linear(1024, 512),
            ReLU(),
            Linear(512, 10)
        )

    def forward(self, batch):
        return self.network(batch)


if __name__ == "__main__":
    from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt
    from sys import stderr
    from os import path
    from random import randint

    BATCH_SIZE = 64
    CLASSES = 10
    INPUTS_VECTOR = 3 * 32 * 32
    TRAINING_ITERATIONS = 2
    STORING_NAME = 'cifar_convolutional_model.pth'
    OPTIMIZER = Adam

    stats = ((0.5, 0.5, 0.4), (0.2, 0.2, 0.2))
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats, inplace=True)
    ])
    validation_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])
    training_dataset = CIFAR10('data/', train=True, transform=train_transform)
    validation_dataset = CIFAR10('data/', train=False, transform=validation_transform)

    classes = training_dataset.classes
    training_dataloader = DeviceDataLoader(DataLoader(training_dataset, BATCH_SIZE, shuffle=True, num_workers=3, pin_memory=True), GenericFunctions.getDefaultProcessingDevice())
    validation_dataloader = DeviceDataLoader(DataLoader(validation_dataset, BATCH_SIZE, num_workers=3, pin_memory=True), GenericFunctions.getDefaultProcessingDevice())
    model = ConvolutionalNetwork2()
    GenericFunctions.toDevice(model, GenericFunctions.getDefaultProcessingDevice())

    histories = []

    trained_model_path = input("Please enter path for trained model (press enter for training): ")
    if path.isfile(trained_model_path):
        model.load_state_dict(torch.load(trained_model_path))
        result0 = GenericFunctions.evaluate(model, validation_dataloader)
    else:
        result0 = GenericFunctions.evaluate(model, validation_dataloader)

    train_model_prompt = input("do you want to continue train the model (y for yes): ")
    print("start model accuracy: {}, loss: {}".format(result0['accuracy_value'], result0['loss_value']))

    if train_model_prompt.lower() == 'y':
        for i in range(TRAINING_ITERATIONS):
            print("Started training model iteration [{}]".format(i))
            history = GenericFunctions.fit(5, 0.0001, model, training_dataloader, validation_dataloader, optimizer_function=OPTIMIZER)
            histories += history
            torch.save(model.state_dict(), STORING_NAME)

        total_accuracy = sorted(
            [result0['accuracy_value']] + [history['accuracy_value'].item() for history in histories])

        plt.plot(total_accuracy, '-x')
        plt.xlabel('iterations')
        plt.ylabel('accuracy')
        plt.title('Accuracy Vs Iterations')
    plt.show(block=False)

    print("[+] trying to allocate testing dataset")
    testing_dataset_images_count = len(validation_dataset)
    print("testing dataset allocated successfully.\n\t number of images: {}".format(testing_dataset_images_count))
    print("evaluating model on testing dataset.")
    testing_dataset_evaluation = GenericFunctions.evaluate(model, validation_dataloader)
    print("testing dataset evaluation accuracy: {}".format(testing_dataset_evaluation['accuracy_value']))

    def randomTests(tests=1000):
        fail, success = [0, 0]
        for i in range(tests):
            image_number = randint(0, testing_dataset_images_count - 1)
            image_tensor, label = validation_dataset[image_number]
            prediction, label = classes[GenericFunctions.predict(image_tensor, model)], classes[label]
            if prediction != label:
                fail += 1
            else:
                success += 1

            print("[testing {}/{}, fail: {}, success: {}] prediction: {}, the true value: {}".format(i + 1, tests, fail,
                                                                                                     success,
                                                                                                     prediction, label))


    randomTests()
    manual_testing_approval = input("-> enter start to start manual testing: ")

    if manual_testing_approval.lower() == "start":
        while True:
            image_number = input("Please enter image number, from 0 to {}, press any letter to terminate: ".format(
                testing_dataset_images_count))
            try:
                image_number = int(image_number)
                if 0 <= image_number < testing_dataset_images_count:
                    image_tensor, label = testing_dataset[image_number]
                    plt.imshow(image_tensor[0], cmap='gray')
                    print("prediction: {}, the true value: {}".format(
                        classes[GenericFunctions.predict(image_tensor, model)],
                        classes[label]))
                    plt.show()
                else:
                    print("Please enter valid number.", file=stderr)
                    continue
            except ValueError:
                break