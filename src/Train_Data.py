""" Train Data and Calculate Loss for each Batch"""


def train(dataloader, model, loss_fn, optim, device):
    model.train(True)
    for step, inputs in enumerate(dataloader):
        inputs = inputs.to(device)
        _, _, decoding = model(inputs.float())
        loss = loss_fn(decoding, inputs.float())
        optim.zero_grad()
        loss.backward()
        optim.step()

        if step % 10 == 0:
            print("Batch : " + str(step) + " Loss : " + str(loss.item()))
