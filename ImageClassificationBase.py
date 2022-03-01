from torch.nn import Linear, ReLU, Module
from torch.nn.functional import cross_entropy
from genericFunctions import GenericFunctions, DeviceDataLoader


class ImageClassificationBase(Module):
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
