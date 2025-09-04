# 8-PSK-BER-Simulation-Neural-Network
Author: Narges Asghari
Type: Bachelor Thesis / Research Project
Technology Stack: Python, NumPy, SciPy, Matplotlib, TensorFlow, scikit-learn

# Overview
This project implements a Bit Error Rate (BER) simulation for an 8-PSK (Phase Shift Keying) digital communication system and extends the analysis using a Neural Network for signal demodulation.

The simulation compares theoretical BER, classical simulated BER, and BER predicted by a neural network under different Eb/N0 (energy per bit to noise power spectral density ratio) conditions.

The goal is to demonstrate the performance of 8PSK modulation in AWGN channels and explore machine learning-based demodulation techniques for communication signals.

# Features

Generates random binary bit streams and maps them to 8PSK symbols using Gray coding.
Adds AWGN (Additive White Gaussian Noise) to simulate realistic channel conditions.
Calculates simulated BER and compares it to theoretical BER.
Implements a custom neural network using TensorFlow to predict transmitted bits from noisy received symbols.
Plots constellation diagrams, BER curves, and training progress of the neural network.
Provides a clear comparison of classical and NN-based BER performance.

# Installation
Clone the repository:
git clone https://github.com/YourUsername/8PSK-BER-Simulation.git
cd 8PSK-BER-Simulation


# Create a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows


# Install required dependencies:
pip install -r requirements.txt

# Usage
Run the main simulation script:
python main.py


The script will:
Generate a random bit stream and map it to 8PSK symbols.
Add AWGN noise for a range of Eb/N0 values.
Demodulate the received symbols and calculate BER.
Train a neural network to predict transmitted bits from noisy symbols.

# Plot:
Constellation of transmitted vs received symbols
BER curves (simulation, theory, neural network)
Neural network training loss and accuracy
Results are also printed in a tabular format using tabulate.

# Project Structure
8PSK-BER-Simulation/
├── main.py           # Full simulation and neural network script

├── requirements.txt  # Python dependencies

├── README.md         # Project description and usage

├── results/          # Optional: save BER results or plots

└── plots/            # Optional: saved figures

# BER Theory Reference

The theoretical BER for 8PSK in an AWGN channel is calculated as:

BER = 1/3*erfc(sqrt(3*SNR)*sin(𝜋/8))

Where erfc is the complementary error function.

# Neural Network Approach
A custom feedforward network is used to learn the mapping from noisy received symbols to transmitted bits.
Input: 8PSK symbols (complex → real & imaginary)
Output: Predicted bits per symbol
Training uses Mean Squared Error (MSE) and Adam optimizer.
Neural network performance is evaluated in terms of BER alongside classical simulation.


# Results
Constellation plots showing the transmitted vs received symbols.
BER vs Eb/N0 curves for:
Classical simulation
Theoretical BER
Neural network prediction
Neural network training loss and accuracy plots.

# References
Proakis, J. G., & Salehi, M. (2008). Digital Communications (5th Edition).
TensorFlow Documentation: https://www.tensorflow.org/
SciPy & NumPy: https://www.scipy.org/


