import matplotlib.pyplot as plt
import numpy as np

from Test_Data import calculate_ers


def M_PSK_constellation(M):
    phases = np.linspace(0, 2 * np.pi, M, endpoint=False)
    x = np.cos(phases)
    y = np.sin(phases)
    plt.plot(x, y, "ro", label=str(M) + "-PSK")


def auto_encoder_constellation(encoder):
    plt.scatter(encoder[:, 0], encoder[:, 1], marker="o", label="Auto-Encoder", color="black")


""" Display Auto-Encoder & M-PSK constellation """


def plot_constellation_Auto_Encoder_M_PSK(encoder, M):
    plt.figure(figsize=(6, 6))
    M_PSK_constellation(M)
    auto_encoder_constellation(encoder)
    plt.title("Constellations M-PSK & Auto-Encoder")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.grid(True)
    plt.legend()
    plt.axis("equal")
    plt.show()


""" Display Error Rates from Auto-Encoder depending on the Eb/N0 ratio """


def plot_er_ebn0(model, dataloader, ebn0_range, device):
    tab_er = calculate_ers(model, dataloader, ebn0_range, device)
    plt.plot(ebn0_range, tab_er, label="Auto-Encoder", color="red")


""" Display Error Rates from Auto-Encoder & M-PSK depending on the Eb/N0 ratio """


def plot_er_ebn0_8psk(model, dataloader, ebn0_range, er_mpsk, ebnodb_mpsk, device):
    plot_er_ebn0(model, dataloader, ebn0_range, device)
    plt.plot(ebnodb_mpsk, er_mpsk, label="M-PSK", color="black")

    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Error Rate")
    plt.title("Comparison of Error Rates for Auto-Encoder & M-PSK")
    plt.grid(True)
    plt.legend()
    plt.show()
