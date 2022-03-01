from torch.utils.data import DataLoader, random_split
from torch.nn import Sequential, ReLU, MaxPool2d, Conv2d, Flatten, Linear, BatchNorm2d, Dropout
from torch.optim import Adam
from genericFunctions import GenericFunctions, DeviceDataLoader
from ImageClassificationBase import ImageClassificationBase
import torch


def conv_block(in_channels, out_channels, kernel_size=3, pool=False):
    layers = [Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1),
              BatchNorm2d(out_channels),
              ReLU(inplace=True)
              ]
    if pool:
        layers.append(MaxPool2d(2))
    return Sequential(*layers)


class RS9(ImageClassificationBase):
    def __init__(self, input_channels=3, output_channels=10, kernel_size=3, intermediate_channels=(16, 32, 64, 128)):
        super(RS9, self).__init__()
        self.layer_1 = conv_block(input_channels, intermediate_channels[0], kernel_size)
        self.layer_2 = conv_block(intermediate_channels[0], intermediate_channels[1], kernel_size, pool=True)
        self.residual_layer1 = Sequential(conv_block(intermediate_channels[1], intermediate_channels[1], kernel_size),
                                          conv_block(intermediate_channels[1], intermediate_channels[1], kernel_size))
        self.layer_3 = conv_block(intermediate_channels[1], intermediate_channels[2], kernel_size, pool=True)
        self.layer_4 = conv_block(intermediate_channels[2], intermediate_channels[3], kernel_size, pool=True)
        self.residual_layer2 = Sequential(conv_block(intermediate_channels[3], intermediate_channels[3], kernel_size),
                                          conv_block(intermediate_channels[3], intermediate_channels[3], kernel_size))

        self.classifier = Sequential(MaxPool2d(4),
                                     Flatten(),
                                     Dropout(p=0.25, inplace=True),
                                     Linear(intermediate_channels[3], output_channels))

    def forward(self, batch):
        out = self.layer_1(batch)
        out = self.layer_2(out)
        out = self.residual_layer1(out) + out
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = self.residual_layer2(out) + out
        return self.classifier(out)


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
    OPTIMIZER = Adam

    # training_dataset = CIFAR10('data/', train=True, transform=transforms.ToTensor())
    # stats = GenericFunctions.calculateMeanAndStd(training_dataset)
    # del training_dataset
    # print(*stats)

    print("[+] making initial data argumentation and normalization.")
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats, inplace=True)
    ])

    print("[+] loading dataset.")
    validation_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])
    training_dataset = CIFAR10('data/', train=True, transform=train_transform)
    validation_dataset = CIFAR10('data/', train=False, transform=validation_transform)

    classes = training_dataset.classes
    training_dataloader = DeviceDataLoader(
        DataLoader(training_dataset, BATCH_SIZE, shuffle=True, num_workers=3, pin_memory=True),
        GenericFunctions.getDefaultProcessingDevice())
    validation_dataloader = DeviceDataLoader(DataLoader(validation_dataset, BATCH_SIZE, num_workers=3, pin_memory=True),
                                             GenericFunctions.getDefaultProcessingDevice())
    model = RS9()
    GenericFunctions.toDevice(model, GenericFunctions.getDefaultProcessingDevice())

    histories = []

    trained_model_path = input("[?] Please enter path for trained model (press enter for training): ")
    if path.isfile(trained_model_path):
        model.load_state_dict(torch.load(trained_model_path))
        print("[+] model parameters loaded from the file successfully.")
    else:
        print("[-] couldn't load model parameters from the file. [-] error: file not exists.")

    print('[+] evaluating initial model accuracy and loss value.')
    result0 = GenericFunctions.evaluate(model, validation_dataloader)
    print("-> initial model accuracy: {}, loss: {}".format(result0['accuracy_value'], result0['loss_value']))
    train_model_prompt = input("do you want to train the model (y for yes): ")

    if train_model_prompt.lower() == 'y':
        print("[+] starting model training.")
        history = GenericFunctions.fitOneCycle(100, 0.001, model, training_dataloader, validation_dataloader, 1e-4, 0.1,
                                               OPTIMIZER, file=trained_model_path)
        histories += history

        total_accuracy = sorted(
            [result0['accuracy_value']] + [history['accuracy_value'].item() for history in histories])

        plt.plot(total_accuracy, '-x')
        plt.xlabel('iterations')
        plt.ylabel('accuracy')
        plt.title('Accuracy Vs Iterations')
        plt.show()

    try:
        model.load_state_dict(torch.load(trained_model_path))
        print("[+] model parameters loaded successfully.")
    except Exception as e:
        print("[-] couldn't load the parameters from {}.\n\t  [-] error: {}".format(trained_model_path, e))
    model.eval()

    print("[+] trying to allocate testing dataset")
    testing_dataset_images_count = len(validation_dataset)
    print("[+] testing dataset allocated successfully.\n\t number of images: {}".format(testing_dataset_images_count))
    print("evaluating model on testing dataset.")
    testing_dataset_evaluation = GenericFunctions.evaluate(model, validation_dataloader)
    print("-> testing dataset evaluation accuracy: {}".format(testing_dataset_evaluation['accuracy_value']))


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

            print("-> [testing {}/{}, fail: {}, success: {}] prediction: {}, the true value: {}".format(i + 1, tests,
                                                                                                        fail,
                                                                                                        success,
                                                                                                        prediction,
                                                                                                        label))


    randomTests()
    manual_testing_approval = input("[?] enter start to start manual testing: ")

    if manual_testing_approval.lower() == "start":
        while True:
            image_number = input("[?] Please enter image number, from 0 to {}, press any letter to terminate: ".format(
                testing_dataset_images_count))
            try:
                image_number = int(image_number)
                if 0 <= image_number < testing_dataset_images_count:
                    image_tensor, label = validation_dataset[image_number]
                    plt.imshow(image_tensor[0], cmap='gray')
                    print("-> prediction: {}, the true value: {}".format(
                        classes[GenericFunctions.predict(image_tensor, model)],
                        classes[label]))
                    plt.show()
                else:
                    print("[-] Please enter valid number.", file=stderr)
                    continue
            except ValueError:
                break
