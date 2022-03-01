if __name__ == "__main__":
    import torch
    from genericFunctions import GenericFunctions, DeviceDataLoader
    from FeedForwardNeuralNetwork import FeedForwardNeuralNetwork
    from logisticRegression import logisticRegressionModel
    from torchvision.datasets import FashionMNIST, CIFAR10
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt
    from sys import stderr
    from os import path
    from random import randint
    from torch.utils.data import DataLoader, random_split

    BATCH_SIZE = 128
    CLASSES = 10
    INPUTS_VECTOR = 3 * 32 * 32
    TRAINING_ITERATIONS = 2
    STORING_NAME = 'fashionMNISET.pth'

    training_dataset = CIFAR10('data/', train=True, transform=transforms.ToTensor(), download=True)
    classes = training_dataset.classes
    training_dataset, validation_dataset = random_split(training_dataset, [40000, 10000])
    validation_dataloader = DeviceDataLoader(DataLoader(validation_dataset, BATCH_SIZE), GenericFunctions.getDefaultProcessingDevice())
    model = FeedForwardNeuralNetwork(INPUTS_VECTOR, CLASSES, training_dataset, batch_size=384)
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
            history = GenericFunctions.fit(100, 0.2, model, model.dataloader, validation_dataloader)
            histories += history

        total_accuracy = sorted(
            [result0['accuracy_value']] + [history['accuracy_value'].item() for history in histories])

        plt.plot(total_accuracy, '-x')
        plt.xlabel('iterations')
        plt.ylabel('accuracy')
        plt.title('Accuracy Vs Iterations')
        plt.show()

    print("[+] trying to allocate testing dataset")
    testing_dataset = CIFAR10("data/", train=False, transform=transforms.ToTensor(), download=True)
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