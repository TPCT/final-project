import torch
import numpy as np
import torch.nn as neural_network
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.functional import mse_loss
from torch.optim import SGD
from random import randint


class GradientDescentModel:
    """
        this is simple model which apply gradient descent and linear regression concepts.
        assume that we have function y that takes the inputs and gives us the outputs
        we want to get the mathematical function that can calculate this values (approximately)
        so we convert it -> y1 = a11 * x1 + a12 * x2 + a13 * x3 + b1
                            where aij is the weight of the given variables.
                                  bi is the bias value of the given function (initial value)
        :param known_inputs -> the inputs you know
        :param expected_outputs -> the outputs you know
    """

    @staticmethod
    def generateRandomData(rows: int, columns: int, min_number: int, max_number: int):
        output = []
        for row_index in range(rows):
            row = [randint(min_number, max_number) for x in range(columns)]
            output.append(row)
        return output

    def __init__(self, known_inputs: np.array, expected_outputs: np.array, learningStep: float):
        self.known_inputs = torch.from_numpy(known_inputs)
        self.target_values = torch.from_numpy(expected_outputs)
        known_inputs_columns = known_inputs.shape[-1]
        expected_outputs_columns = expected_outputs.shape[-1]
        self.weights = torch.randn(expected_outputs_columns, known_inputs_columns, requires_grad=True)
        self.bias = torch.randn(expected_outputs_columns, requires_grad=True)
        self.learningStep = learningStep

    def functionGenerator(self):
        return self.known_inputs @ self.weights.t() + self.bias

    def lossFunction(self):
        predictions = self.functionGenerator()
        losses = predictions - self.target_values
        losses_squared = losses * losses
        mean_square_losses = torch.sum(losses_squared) / losses.numel()
        mean_square_losses.backward()
        return mean_square_losses

    def correctionFunction(self):
        with torch.no_grad():
            self.weights -= self.weights.grad * self.learningStep
            self.bias -= self.bias.grad * self.learningStep
            self.weights.grad.zero_()
            self.bias.grad.zero_()
            return {
                'weights': self.weights,
                'bias': self.bias
            }

    def correct(self, iterations=100000):
        print(f"[+] starting with loss function: {self.lossFunction()}")
        min_lossFunction = None
        corrected_weights = None
        for i in range(iterations):
            min_lossFunction = self.lossFunction()
            corrected_weights = self.correctionFunction()
        print(f"[+] ending with loss function: {min_lossFunction} after {iterations} iterations")
        return min_lossFunction, corrected_weights


if __name__ == "__main__":
    BATCH_SIZE = 100
    inputs = np.array(GradientDescentModel.generateRandomData(1000, 3, 10, 100), dtype="float32")
    targets = np.array(GradientDescentModel.generateRandomData(1000, 2, 50, 100), dtype="float32")
    inputs = torch.from_numpy(inputs)
    targets = torch.from_numpy(targets)

    train_ds = TensorDataset(inputs, targets)
    train_dl = DataLoader(train_ds, BATCH_SIZE, True)

    model = neural_network.Linear(3, 2)
    predictions = model(inputs)
