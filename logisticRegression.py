import torch
from torch.nn import Linear, Module
from torch.nn.functional import cross_entropy
from torch.utils.data import random_split, DataLoader
from genericFunctions import GenericFunctions


class logisticRegressionModel(Module):
    def __init__(self, in_features_size, out_features_size, dataset):
        super().__init__()
        self.linear_model = Linear(in_features_size, out_features_size)
        self.in_features_size = in_features_size
        self.out_features_size = out_features_size
        self.dataset = dataset
        self.dataloader = None

    def generateDataLoader(self, batch_size):
        self.dataloader = DataLoader(self.dataset, batch_size)

    def forward(self, batch):
        batch = batch.reshape(-1, self.in_features_size)
        return self.linear_model(batch)

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
    from torchvision.datasets import MNIST
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt
    from threading import Thread, Lock, currentThread
    from sys import stderr
    from os import path
    from random import randint

    BATCH_SIZE = 128
    CLASSES = 10
    INPUTS_VECTOR = 28 * 28
    TRAINING_ITERATIONS = 2
    STORING_NAME = 'model1.pth'
    THREADS_LOCK = Lock()

    training_dataset = MNIST('data/', train=True, transform=transforms.ToTensor())
    training_dataset, validation_dataset = random_split(training_dataset, [50000, 10000])
    validation_dataloader = DataLoader(validation_dataset, BATCH_SIZE)
    model = logisticRegressionModel(INPUTS_VECTOR, CLASSES, training_dataset)
    model.generateDataLoader(BATCH_SIZE)

    histories = {}
    threads_pool = []

    def createHistoryThread(pool, target, *args, **kwargs):
        while True:
            try:
                thread_name = "{}".format(len(pool))
                histories[thread_name] = []
                thread = Thread(name=thread_name, target=target, args=args, kwargs=kwargs)
                thread.start()
                pool.append(thread)
                break
            except RuntimeError:
                continue

    def historyThread(result_saver, iterations, learning_step, model, training_dataloader, validation_dataloader, lock=THREADS_LOCK):
        history = GenericFunctions.fit(iterations, learning_step, model, training_dataloader, validation_dataloader, lock=lock)
        with lock:
            result_saver[currentThread().name] += history

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
            createHistoryThread(threads_pool, historyThread, histories, 5, 0.1, model, model.dataloader,
                                validation_dataloader, THREADS_LOCK)

        for thread in threads_pool:
            thread.join()
        temp_histories = []
        histories_keys = sorted(list(histories.keys()))
        for key in histories_keys:
            temp_histories += histories[key]
        histories = temp_histories
        total_accuracy = sorted(
            [result0['accuracy_value']] + [history['accuracy_value'].item() for history in histories])

        plt.plot(total_accuracy, '-x')
        plt.xlabel('iterations')
        plt.ylabel('accuracy')
        plt.title('Accuracy Vs Iterations')
        plt.show()

    print("[+] trying to allocate testing dataset")
    testing_dataset = MNIST("data/", train=False, transform=transforms.ToTensor(), download=True)
    testing_dataloader = DataLoader(testing_dataset, BATCH_SIZE)
    testing_dataset_images_count = len(testing_dataset)
    print("testing dataset allocated successfully.\n\t number of images: {}".format(testing_dataset_images_count))
    print("evaluating model on testing dataset.")
    testing_dataset_evaluation = GenericFunctions.evaluate(model, testing_dataloader)
    print("testing dataset evaluation accuracy: {}".format(testing_dataset_evaluation['accuracy_value']))

    torch.save(model.state_dict(), STORING_NAME)

    def randomTests(tests=1000):
        fail, success = [0, 0]
        for i in range(tests):
            image_number = randint(0, testing_dataset_images_count-1)
            image_tensor, label = testing_dataset[image_number]
            prediction, label = GenericFunctions.predict(image_tensor, model), label
            if prediction != label:
                fail += 1
            else:
                success += 1

            print("[testing {}/{}, fail: {}, success: {}] prediction: {}, the true value: {}".format(i+1, tests, fail, success, prediction, label))

    randomTests()
    manual_testing_approval = input("-> enter start to start manual testing: ")

    if manual_testing_approval.lower() == "start":
        while True:
            image_number = input("Please enter image number, from 0 to {}, press any letter to terminate: ".format(testing_dataset_images_count))
            try:
                image_number = int(image_number)
                if 0 <= image_number < testing_dataset_images_count:
                    image_tensor, label = testing_dataset[image_number]
                    plt.imshow(image_tensor[0], cmap='gray')
                    print("prediction: {}, the true value: {}".format(GenericFunctions.predict(image_tensor, model), label))
                    plt.show()
                else:
                    print("Please enter valid number.", file=stderr)
                    continue
            except ValueError:
                break

