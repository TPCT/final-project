from torch.utils.data import DataLoader, random_split
from torch.nn import Linear, ReLU, Module
from torch.nn.functional import cross_entropy
from genericFunctions import GenericFunctions, DeviceDataLoader
import torch


class FeedForwardNeuralNetwork(Module):
    def __init__(self, in_features_size, out_features_size, dataset, activation_function=ReLU(), intermediate_vector_lengths=(128, 256, 512)):
        super(FeedForwardNeuralNetwork, self).__init__()
        self.layer_1 = Linear(in_features_size, intermediate_vector_lengths[0])
        self.layer_2 = Linear(intermediate_vector_lengths[0], intermediate_vector_lengths[1])
        self.layer_3 = Linear(intermediate_vector_lengths[1], intermediate_vector_lengths[2])
        self.layer_4 = Linear(intermediate_vector_lengths[2], out_features_size)
        self.in_features_size = in_features_size
        self.out_features_size = out_features_size
        self.dataset = dataset
        self.activation_function = activation_function

    def forward(self, batch):
        batch = batch.reshape(-1, self.in_features_size)
        layer1_outputs = self.activation_function(self.layer_1(batch))
        layer2_outputs = self.activation_function(self.layer_2(layer1_outputs))
        layer3_outputs = self.activation_function(self.layer_3(layer2_outputs))
        outputs = self.layer_4(layer3_outputs)
        return outputs

    def trainingStep(self, batch):
        images, labels = batch
        predictions = self(images)
        loss = cross_entropy(predictions, labels)
        return loss

    def validationStep(self, batch):
        images, labels = batch
        predictions = self(images)
        loss = cross_entropy(predictions, labels)
        accuracy = GenericFunctions.accuracy(predictions, labels)
        return {
            'loss_value': loss,
            'accuracy_value': accuracy
        }


if __name__ == "__main__":
    from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt
    from sys import stderr
    from os import path
    from random import randint

    BATCH_SIZE = 128
    CLASSES = 10
    INPUTS_VECTOR = 28 * 28
    TRAINING_ITERATIONS = 2
    STORING_NAME = 'model2.pth'

    training_dataset = FashionMNIST('data/', train=True, transform=transforms.ToTensor(), download=True)
    classes = training_dataset.classes
    training_dataset, validation_dataset = random_split(training_dataset, [50000, 10000])
    validation_dataloader = DeviceDataLoader(DataLoader(validation_dataset, BATCH_SIZE), GenericFunctions.getDefaultProcessingDevice())
    model = FeedForwardNeuralNetwork(INPUTS_VECTOR, CLASSES, training_dataset)
    GenericFunctions.toDevice(model, GenericFunctions.getDefaultProcessingDevice())

    histories = []

    trained_model_path = input("Please enter path for trained model (press enter for training): ")
    if path.isfile(trained_model_path):
        model.load_state_dict(torch.load(trained_model_path))
        result0 = GenericFunctions.evaluate(model, validation_dataloader)
        print("start model accuracy: {}, loss: {}".format(result0['accuracy_value'], result0['loss_value']))
    else:
        result0 = GenericFunctions.evaluate(model, validation_dataloader)
        print("start model accuracy: {}, loss: {}".format(result0['accuracy_value'], result0['loss_value']))
        for i in range(TRAINING_ITERATIONS):
            print("Started training model iteration [{}]".format(i))
            history = GenericFunctions.fit(5, 0.01, model, model.dataloader, validation_dataloader)
            histories += history

        total_accuracy = sorted(
            [result0['accuracy_value']] + [history['accuracy_value'].item() for history in histories])

        plt.plot(total_accuracy, '-x')
        plt.xlabel('iterations')
        plt.ylabel('accuracy')
        plt.title('Accuracy Vs Iterations')
        plt.show()

    print("[+] trying to allocate testing dataset")
    testing_dataset = FashionMNIST("data/", train=False, transform=transforms.ToTensor(), download=True)
    testing_dataloader = DataLoader(testing_dataset, BATCH_SIZE)
    testing_dataloader = DeviceDataLoader(testing_dataloader, GenericFunctions.getDefaultProcessingDevice())
    testing_dataset_images_count = len(testing_dataset)
    print("testing dataset allocated successfully.\n\t number of images: {}".format(testing_dataset_images_count))
    print("evaluating model on testing dataset.")
    testing_dataset_evaluation = GenericFunctions.evaluate(model, testing_dataloader)
    print("testing dataset evaluation accuracy: {}".format(testing_dataset_evaluation['accuracy_value']))

    torch.save(model.state_dict(), STORING_NAME)


    def randomTests(tests=1000):
        fail, success = [0, 0]
        for i in range(tests):
            image_number = randint(0, testing_dataset_images_count - 1)
            image_tensor, label = testing_dataset[image_number]
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
                    print("prediction: {}, the true value: {}".format(classes[GenericFunctions.predict(image_tensor, model)],
                                                                      classes[label]))
                    plt.show()
                else:
                    print("Please enter valid number.", file=stderr)
                    continue
            except ValueError:
                break
