import torch
from math import e

def exponentialSum(vector):
    accumulator = 0
    for element in vector:
        accumulator += e ** element
    return accumulator


def softmax(input_vector):
    for i in range(input_vector.shape[0]):
        output = torch.zeros(input_vector.shape)

        if len(input_vector.shape) == 1:
            row_exp_sum = exponentialSum(input_vector)
        else:
            row_exp_sum = exponentialSum(input_vector[i])

        for j in range(input_vector.shape[-1]):
            if len(input_vector.shape) == 1:
                output[j] = ((e ** input_vector[j]) / row_exp_sum)
            else:
                output[i][j] = ((e ** input_vector[i][j]) / row_exp_sum)

        return output


def applyKernel(image, kernel):
    image_rows, image_columns = image.shape
    kernel_rows, kernel_columns = kernel.shape
    output_matrix_rows, output_matrix_columns = image_rows - kernel_rows + 1, image_columns - kernel_columns + 1
    output = torch.zeros([output_matrix_rows, output_matrix_columns])
    for i in range(output_matrix_rows):
        for j in range(output_matrix_columns):
            output[i, j] = torch.sum(image[i:i+kernel_rows, j:j+kernel_columns] * kernel)
    return output