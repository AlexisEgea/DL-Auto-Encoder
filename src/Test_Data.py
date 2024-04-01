import numpy as np
import torch

from Variable.variable import debug

""" Test & Calculate Accuracy of Auto-Encoder
    Returns the most probable value of the estimated message (ŝ) by the Auto-Encoder. """


def test(dataloader, model, device):
    correct = 0
    total = 0
    most_probable_values = []
    model.eval()
    with torch.no_grad():
        for inputs in dataloader:
            inputs = inputs.to(device)
            _, _, decoding = model(inputs.float())
            predicted = decoding.argmax(1)
            inputs = inputs.argmax(1)
            total += inputs.size(0)
            correct += (predicted == inputs).sum().item()

            most_probable_value = torch.mode(predicted)[0]
            most_probable_values.append(most_probable_value.item())

    accuracy = 100 * correct // total
    print("Accuracy : " + str(accuracy) + "%")

    # Calculer la valeur la plus fréquente dans l'ensemble des données
    final_most_probable_value = torch.mode(torch.tensor(most_probable_values))[0].item()
    print("Most Probable Value of the Estimated Message (ŝ)= " + str(final_most_probable_value))
    return final_most_probable_value


""" Calculate Error Rate ((Difference between Input & Predicted Output) / (Total Input)) of Auto-Encoder depending on 
the Eb/N0 ratio """


def calculate_er(model, test_dataloader, EbN0_dB, device):
    model.eval()
    EbN0 = 10 ** (EbN0_dB / 10.0)
    noise_EbN0 = np.sqrt(1 / (2 * EbN0))

    total = 0
    errors = 0

    with torch.no_grad():
        for inputs in test_dataloader:
            inputs = inputs.to(device)
            _, _, decoding = model(inputs.float())

            noise = noise_EbN0 * torch.randn_like(decoding)
            predicted = decoding + noise

            predicted = predicted.argmax(dim=1)
            inputs = inputs.argmax(dim=1)
            total += inputs.size(0)
            errors += (predicted != inputs).sum().item()

    er = errors / total

    return er


""" Return Error Rates of Auto-Encoder for each Eb/N0 range  """


def calculate_ers(model, test_dataloader, ebn0_range, device):
    if debug:
        print("Error Rate")
    bers = []
    for ebn0 in ebn0_range:
        ber = calculate_er(model, test_dataloader, ebn0, device)
        if debug:
            print(ber)
        bers.append(ber)
    return bers
