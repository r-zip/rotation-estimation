import torch
from ../script import data.py

def trainModel(dataloader, model, loss_fn, optimizer, device):
    """
    Function that trains the model for one batch and reports summary statistics

    :param dataloader:
    :param model:
    :
    """

    size = len(dataloader.dataset)
    model.train()
    model_history = []
    averageLoss = 0

    for batch, (x, y) in enumerate(dataloader):
        # Loop through each batch and backprop through model

        x = x.to(device)
        y = y.to(device)

        #prediction and loss
        pred = model(x)
        # print(pred.size())
        # print(y)
        loss = loss_fn(pred,y)
        lOne = lOnePenalty*sum(torch.abs(p).sum() for p in model.parameters())
        lossR = loss + lOne

        # Backpropagation
        optimizer.zero_grad()
        lossR.backward()
        optimizer.step()

        averageLoss = averageLoss + loss.item()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(x)
            model_history.append(loss)
            print(f"Loss: {loss:>7f} Epoch average loss: {averageLoss/(batch+1):>7f} [{current:>5d}/{size:>5d}]")
            print(f"L1 Regularization term: {lOne.item()}")

            # Only consider subset of the entire dataset to speed up training during pretraining
            if current/size > considerPercent:
                break
    return model_history