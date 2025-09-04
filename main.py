import numpy as np
import matplotlib.pyplot as plt
from random import randint
from random import seed
from scipy.special import erfc  # erfc/Q function
import array
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.keras.layers import *
from tabulate import tabulate as table

# -------------------- Generating Bits ------------------------------

Bits = 10000
num_sym = (Bits + (3 - Bits % 3))
tx = np.random.randint(2, size=num_sym)
Tx = tx.reshape(int(num_sym / 3), 3)
print("Tx = ", Tx)
BER_Simulation = []
BER_theory = []
#         # ------------- Plot the Signals ----------------
plt.figure(figsize=(9, 3))
plt.stem(tx) # Removed use_line_collection=True
plt.title("Random Bits (100 Bits)")
plt.grid(True)
plt.legend("Random num ", loc="upper right")
# plt.show()


# --------------------- Modulation ---------------------------------
mod = []
map8Psk = [[complex(np.cos(np.pi / 8), np.sin(np.pi / 8))],
           [complex(np.cos(3 * np.pi / 8), np.sin(3 * np.pi / 8))],
           [complex(np.cos(5 * np.pi / 8), np.sin(5 * np.pi / 8))],
           [complex(np.cos(7 * np.pi / 8), np.sin(7 * np.pi / 8))],
           [complex(np.cos(9 * np.pi / 8), np.sin(9 * np.pi / 8))],
           [complex(np.cos(11 * np.pi / 8), np.sin(11 * np.pi / 8))],
           [complex(np.cos(13 * np.pi / 8), np.sin(13 * np.pi / 8))],
           [complex(np.cos(15 * np.pi / 8), np.sin(15 * np.pi / 8))]]
Map8psk = np.array(map8Psk)
graycode = [[0, 0, 0], [0, 0, 1], [0, 1, 1],
            [0, 1, 0], [1, 1, 0], [1, 1, 1],
            [1, 0, 1], [1, 0, 0]]
Graycode = np.array(graycode)

for i in range(len(Tx)):
    iNdex = np.where(np.all(Tx[i,] == Graycode, axis=1))
    mod.append(Map8psk[iNdex[0]])
Symbol_mod = np.array(mod)
lis_Symbol = np.squeeze(Symbol_mod)


# ------------------------- Channel ---------------------------------
# Compute power in modulatedSyms and add AWGN noise for given SNRs
EbN0dBs = np.arange(start=-5, stop=10)  # Eb/N0 range in dB for simulation
for k, EbN0dB in enumerate(EbN0dBs):
    gamma = 10 ** (EbN0dB / 10)  # SNRs to linear scale
    P = sum(abs(Symbol_mod) ** 2) / len(Symbol_mod)  # Actual power in the vector
    N0 = P / gamma  # Find the noise spectral density
    A = 1 / np.sqrt(2)
    II = np.random.standard_normal(Symbol_mod.shape)
    Q = np.random.standard_normal(Symbol_mod.shape)
    sigma = np.sqrt(1 / (np.log2(8) * gamma))
    Noise = A * (Q + 1j * II)
    Rx_sym = Symbol_mod + Noise * sigma  # received signal

    # -------------------------- Demodulation ---------------------------
    Symbol_Matrix = np.tile(np.squeeze(Rx_sym, 1), (1, 8))  # [33334*8]
    Map_Matrix = np.tile(Map8psk.T, (int(num_sym / 3), 1))  # [33334*8]
    data_Matrix = abs(Symbol_Matrix - Map_Matrix)
    Symbol_dmod = np.argmin(data_Matrix, axis=1)

    #  ------------------------- Decimal to Binary ---------------------
    Rx_bit = []
    for i in range(len(Symbol_dmod)):
        Rx_bit.append(Graycode[Symbol_dmod[i]])
    Rx = np.squeeze(np.array(Rx_bit).reshape(1, num_sym), 0)

    #  ------------------------ BER Calculation ------------------------

    Errors = np.sum(abs(Rx - tx))
    print(Errors)
    BER_Simulation.append([abs(Errors / Bits)])
    BER_theory.append(1 / 3 * erfc((np.sqrt(3 * gamma) * np.sin(np.pi / 8))))  # theoretical ber of 8PSK

#  --------------------------- plot ---------------------------------
EbN0dBs = np.arange(start=-5, stop=10)  # Eb/N0 range in dB for simulation
fig, ax1 = plt.subplots()
x = Map8psk.real;     y = Map8psk.imag
x2 = Rx_sym.real;     y2 = Rx_sym.imag
x3 = lis_Symbol.real; y3 = lis_Symbol.imag
# s3 = ax1.scatter(x3, y3)
s1 = ax1.scatter(x2, y2)
s2 = ax1.scatter(x, y, color="Magenta", marker="X", linewidths=5)
ax1.set_xlabel('In phase Component')
ax1.set_ylabel('Quadrature Component')
ax1.set_title('Constellation of Transmitted Symbols Vs Received')
ax1.legend((s1, s2), ("Rx_Symbol", "Tx_Symbol"), loc="upper right")

fig2, ax = plt.subplots()
d1 = ax.semilogy(EbN0dBs, BER_Simulation, color='r', marker='o')
d2 = ax.semilogy(EbN0dBs, BER_theory, marker='*', color='y', label='8PSK Theory')
ax.legend(['BER_Simulation', 'BER_theory'])
ax.set_xlabel('EbN0dbs')
ax.set_ylabel('BER')
ax.legend((d1, d2), ("Simulation", "Theory"), loc="upper right")
ax.set_title('BER Simulation Vs Theory')
ax.grid(True)
plt.show()


# --------------------------- Neural Network --------------------------------------------------------------------------

def Ber_N(YYY):
    factor = 1
    Errors_NN = np.sum(abs(YYY - Tx) * factor)
    print(Errors_NN)
    return Errors_NN


def trans_Vrf(temp):
    v_real = tf.cos(temp)
    v_imag = tf.sin(temp)
    vrf = tf.cast(tf.complex(v_real, v_imag), tf.complex128)
    return vrf


#
#
# ======================================================================================================================
#                                       Creat Network
# ======================================================================================================================

# Generate Bit and data for a specific EbN0dB
# ===========================================
def Generate_Data_Single_EbN0(Bits, EbN0dB):
    num_sym1 = (Bits + (3 - Bits % 3))
    tx1 = np.random.randint(2, size=num_sym1)
    Tx1 = tx1.reshape(int(num_sym1 / 3), 3)
    mod1 = []

    Map8psk1 = np.array(map8Psk)
    graycode = [[0, 0, 0], [0, 0, 1], [0, 1, 1],
                [0, 1, 0], [1, 1, 0], [1, 1, 1],
                [1, 0, 1], [1, 0, 0]]
    Graycode1 = np.array(graycode)
    for i in range(len(Tx1)):
        iNdex1 = np.where(np.all(Tx1[i,] == Graycode1, axis=1))
        mod1.append(Map8psk1[iNdex1[0]])
    Symbol_mod1 = np.array(mod1)
    lis_Symbol1 = np.squeeze(Symbol_mod1)

    # Channel
    # =======
    gamma = 10 ** (EbN0dB / 10)  # SNRs to linear scale
    P1 = sum(abs(Symbol_mod1) ** 2) / len(Symbol_mod1)  # Actual power in the vector
    N01 = P1 / gamma  # Find the noise spectral density
    A = 1 / np.sqrt(2)
    II = np.random.standard_normal(Symbol_mod1.shape)
    Q = np.random.standard_normal(Symbol_mod1.shape)
    sigma = np.sqrt(1 / (np.log2(8) * gamma))
    Noise = A * (Q + 1j * II)
    Rx_sym1 = Symbol_mod1 + Noise * sigma  # received signal

    # -------------------------- Demodulation ---------------------------
    Symbol_Matrix1 = np.tile(np.squeeze(Rx_sym1, 1), (1, 8))  # [3334*8]
    Map_Matrix1 = np.tile(Map8psk1.T, (int(num_sym1 / 3), 1))  # [3334*8]
    data_Matrix1 = abs(Symbol_Matrix1 - Map_Matrix1)
    Symbol_dmod1 = np.argmin(data_Matrix1, axis=1)

    #  ------------------------- Decimal to Binary ---------------------
    Rx_bit1 = []
    for i in range(len(Symbol_dmod1)):
        Rx_bit1.append(Graycode1[Symbol_dmod1[i]])
    Rx = np.squeeze(np.array(Rx_bit1).reshape(1, num_sym1), 0)

    return Rx, Tx1 # Return Tx1 which is the correctly shaped target data


# Generate data for training at a specific EbN0dB (e.g., 5 dB)
train_EbN0dB = 5
data_RX_train, data_TX_train = Generate_Data_Single_EbN0(Bits, train_EbN0dB)


class MyModel(tf.keras.layers.Layer):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(3)
        self.dense2 = tf.keras.layers.Dense(50)
        self.dense3 = tf.keras.layers.Dense(100)
        self.dense4 = tf.keras.layers.Dense(200)
        self.dense5 = tf.keras.layers.Dense(50)
        self.dense6 = tf.keras.layers.Dense(3)

    def call(self, inputs):
        X = self.dense1(inputs)
        X = self.dense2(X)
        X = self.dense3(X)
        X = self.dense4(X)
        X = self.dense5(X)
        X = self.dense6(X)

        return X


model = tf.keras.models.Sequential([
    MyModel()
])

Loss = tf.keras.losses.MeanSquaredError(name="mean_squared_error")
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
model.compile(optimizer=optimizer, loss=Loss, metrics=["accuracy"])

# Train the model
history = model.fit(data_RX_train.reshape((int(num_sym/3), 3)), data_TX_train, validation_split=0.33, epochs=500)
print(history.history.keys())

# Evaluate the trained model over the range of EbN0dBs
BER_Simulation_NN = []
for k, EbN0dB in enumerate(EbN0dBs):
    # Generate data for evaluation at the current EbN0dB
    Rx_N, Tx_N = Generate_Data_Single_EbN0(Bits, EbN0dB)

    #  ------------------------ BER Calculation ------------------------
    pred = np.abs(model.predict(Rx_N.reshape(int(num_sym/3), 3)))
    YYY = pred
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            if pred[i, j] >= 0.5:
                YYY[i, j] = 1
            else:
                YYY[i, j] = 0
    YYY = YYY.astype('int32')

    Err = Ber_N(YYY)
    BER_Simulation_NN.append([abs(Err / Bits)])

# --------------------------- Plot -------------------
# summarize history for accuracy
plt.subplot(211)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
# summarize history for loss
plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

fig3, ax2 = plt.subplots()
d1 = ax2.semilogy(EbN0dBs, BER_Simulation, color='r', marker='o')
d2 = ax2.semilogy(EbN0dBs, BER_theory, marker='*', color='y', label='8PSK Theory')
d3 = ax2.semilogy(EbN0dBs, BER_Simulation_NN, marker='+', color='b', label='8PSK NN')
ax2.legend(['BER_Simulation', 'BER_theory', 'BER_NN'])
ax2.set_xlabel('In phase Component')
ax2.set_ylabel('Quadrature Component')
ax2.set_title('Constellation of Transmitted Symbols Vs Received')
ax2.set_xlabel('EbN0dbs')
ax2.set_ylabel('BER')
# ax.legend((d1, d2), ("Simulation", "Theory"), loc="upper right")
ax2.set_title('BER Simulation Vs Theory')
ax2.grid()
plt.show()

info = {'Eb/N0db': EbN0dBs,
        'BER Theory': BER_theory,
        'BER Simulation': BER_Simulation,
        'BER Simulation_NN': BER_Simulation_NN}

print(table(info, headers='keys', tablefmt="mixed_grid"))
