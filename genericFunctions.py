import torch


class GenericFunctions:
    @staticmethod
    def accuracy(outputs, targets):
        _, predictions = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(predictions == targets).item() / len(predictions))

    @staticmethod
    def getLearningRate(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    @staticmethod
    def calculateMeanAndStd(images, channels=3):
        mean = torch.zeros(1, 3)
        standard_deviation = torch.zeros([1, 3])
        for image, label in images:
            image = image.reshape(-1, image.shape[-1] * image.shape[-2])
            for i in range(channels):
                mean[0][i] += image[i].mean().item()
                standard_deviation[0][i] += image[i].std()
        mean /= len(images)
        standard_deviation /= len(images)

        return mean[0], standard_deviation[0]

    @staticmethod
    def fitOneCycle(iterations, max_learning_rate, model, training_loader, validation_loader, weight_decay=0,
                    gradient_clip=None, optimizer_function=torch.optim.SGD, file=None):
        torch.cuda.empty_cache()
        history = []
        max_accuracy = -1 * torch.inf
        max_parameters = None

        optimizer = optimizer_function(model.parameters(), max_learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_learning_rate, epochs=iterations,
                                                        steps_per_epoch=len(training_loader))

        for iteration in range(iterations):
            model.train()
            train_losses = []
            learning_rates = []
            for batch in training_loader:
                loss = model.trainingStep(batch)
                train_losses.append(loss)
                loss.backward()

                if gradient_clip:
                    torch.nn.utils.clip_grad_value_(model.parameters(), gradient_clip)

                optimizer.step()
                optimizer.zero_grad()

                learning_rates.append(GenericFunctions.getLearningRate(optimizer))
                scheduler.step()

            result = GenericFunctions.evaluate(model, validation_loader)
            result['train_loss'] = torch.stack(train_losses).mean().item()
            result['learning_rates'] = learning_rates

            if file:
                GenericFunctions.iterationEnd(iteration, result)
                if result['accuracy_value'] >= max_accuracy:
                    try:
                        torch.save(model.state_dict(), file)
                        print("\t[+] model parameters saved to {}.".format(file))
                    except Exception as e:
                        print("\t[-] couldn't save model parameters.\n\t\t [-] error: {}".format(e))
                    max_accuracy = result['accuracy_value']

            history.append(result)
        return history

    @staticmethod
    def fit(iterations, learning_rate, model, train_loader, validation_loader, optimizer_function=torch.optim.SGD,
            lock=None):
        history = []
        optimizer = optimizer_function(model.parameters(), learning_rate)

        for iteration in range(iterations):
            model.train()
            train_losses = []
            for batch in train_loader:
                loss = model.trainingStep(batch)
                train_losses.append(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            result = GenericFunctions.evaluate(model, validation_loader)
            result['train_loss'] = torch.stack(train_losses).mean().item()
            GenericFunctions.iterationEnd(iteration, result)
            history.append(result)
        return history

    @staticmethod
    @torch.no_grad()
    def evaluate(model, val_loader):
        model.eval()
        outputs = [model.validationStep(batch) for batch in val_loader]
        return GenericFunctions.validationIterationEnd(outputs)

    @staticmethod
    def validationIterationEnd(outputs):
        batch_losses = [x['loss_value'] for x in outputs]
        iteration_loss = torch.stack(batch_losses).mean()
        batch_accuracy = [x['accuracy_value'] for x in outputs]
        iteration_accuracy = torch.stack(batch_accuracy).mean()
        return {
            'loss_value': iteration_loss,
            'accuracy_value': iteration_accuracy
        }

    @staticmethod
    def iterationEnd(iteration, result):
        print("-> iteration [{}], average losses: {:.4f}, average accuracy: {:.4f}".format(iteration,
                                                                                           result['loss_value'],
                                                                                           result['accuracy_value']))

    @staticmethod
    def predict(image, model):
        image = GenericFunctions.toDevice(image.unsqueeze(0), GenericFunctions.getDefaultProcessingDevice())
        output_prediction = model(image)
        _, prediction = torch.max(output_prediction, dim=1)
        return prediction[0].item()

    @staticmethod
    def getDefaultProcessingDevice():
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    @staticmethod
    def toDevice(data, device):
        if isinstance(data, (list, tuple)):
            return [GenericFunctions.toDevice(x, device) for x in data]
        return data.to(device, non_blocking=True)

    @staticmethod
    def denormalize(images, means, stds):
        means = torch.tensor(means).reshape(1, 3, 1, 1)
        stds = torch.tensor(stds).reshape(1, 3, 1, 1)
        return images * stds + means


class DeviceDataLoader:
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device

    def __iter__(self):
        for batch in self.dataloader:
            yield GenericFunctions.toDevice(batch, self.device)

    def __len__(self):
        return len(self.dataloader)
