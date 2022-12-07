import torch

def trainModel(dataloader, model, loss_fn, optimizer, device):
    """
    Function that trains the model for one epoch and reports summary statistics

    :param dataloader: Pytorch dataloader object that returns batched pairs
    :param model: Pytorch nn.module with forward function
    :param loss_fn: Pytorch loss function
    :param optimizer: Pytorch optimizer object
    :param device: String that specifies the device to run all of the operations on
    :return model_history: A list of floats of the training losses 
    :return averageLoss: A float representing average loss 
    """

    # Initialize variables and model
    size = len(dataloader.dataset)
    model.train()
    model_history = []
    averageLoss = 0

    for batch, (x, y) in enumerate(dataloader):
        # For loop that loops through the entire dataloader. For each iteration,
        # one batch is fed through the model and the loss backpropagated

        # Move batch to device
        x = x.to(device)
        y = y.to(device)

        # Calcualte predictions and loss
        pred = model(x)
        loss = loss_fn(pred,y)

        # Backpropagate through the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Add loss for epoch-wise average loss summary statistic
        averageLoss += loss.item()

        if batch % 100 == 0:
            # Every n-th batch we get summary statistics 
            loss, current = loss.item(), batch * len(x)
            model_history.append(loss)

            print(f"Loss: {loss:>7f} Epoch average loss: {averageLoss/(batch+1):>7f} [{current:>5d}/{size:>5d}]")
    return model_history, averageLoss/size

def testModel(testloader, model, loss_fn, device):
    """
    Function that tests the model

    :param testloader: Pytorch dataloader object that returns batched pairs
    :param model: Pytorch nn.module with forward function
    :param loss_fn: Pytorch loss function
    :param device: string that specifies the device to run all of the operations on
    :return test_loss: loss over the entire batch
    """

    # Initialize variables and model
    size = len(testloader.dataset)
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for batch, (x, y) in enumerate(testloader):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
    test_loss /= size
    print(f"Test Error: \n Avg loss: {test_loss:>8f}")
    return test_loss