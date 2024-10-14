from qiskit import QuantumCircuit, QuantumRegister
import numpy as np

def supremacy_49():
    qc = QuantumCircuit()

    q = QuantumRegister(49, 'q')

    qc.add_register(q)

    qc.h(q[0])
    qc.h(q[1])
    qc.h(q[2])
    qc.h(q[3])
    qc.h(q[4])
    qc.h(q[5])
    qc.h(q[6])
    qc.h(q[7])
    qc.h(q[8])
    qc.h(q[9])
    qc.h(q[10])
    qc.h(q[11])
    qc.h(q[12])
    qc.h(q[13])
    qc.h(q[14])
    qc.h(q[15])
    qc.h(q[16])
    qc.h(q[17])
    qc.h(q[18])
    qc.h(q[19])
    qc.h(q[20])
    qc.h(q[21])
    qc.h(q[22])
    qc.h(q[23])
    qc.h(q[24])
    qc.h(q[25])
    qc.h(q[26])
    qc.h(q[27])
    qc.h(q[28])
    qc.h(q[29])
    qc.h(q[30])
    qc.h(q[31])
    qc.h(q[32])
    qc.h(q[33])
    qc.h(q[34])
    qc.h(q[35])
    qc.h(q[36])
    qc.h(q[37])
    qc.h(q[38])
    qc.h(q[39])
    qc.h(q[40])
    qc.h(q[41])
    qc.h(q[42])
    qc.h(q[43])
    qc.h(q[44])
    qc.h(q[45])
    qc.h(q[46])
    qc.h(q[47])
    qc.h(q[48])
    qc.cz(q[0], q[1])
    qc.t(q[2])
    qc.t(q[3])
    qc.cz(q[4], q[5])
    qc.t(q[6])
    qc.t(q[7])
    qc.t(q[8])
    qc.cz(q[9], q[10])
    qc.t(q[11])
    qc.t(q[12])
    qc.t(q[13])
    qc.cz(q[14], q[15])
    qc.t(q[16])
    qc.t(q[17])
    qc.cz(q[18], q[19])
    qc.t(q[20])
    qc.t(q[21])
    qc.t(q[22])
    qc.cz(q[23], q[24])
    qc.t(q[25])
    qc.t(q[26])
    qc.t(q[27])
    qc.cz(q[28], q[29])
    qc.t(q[30])
    qc.t(q[31])
    qc.cz(q[32], q[33])
    qc.t(q[34])
    qc.t(q[35])
    qc.t(q[36])
    qc.cz(q[37], q[38])
    qc.t(q[39])
    qc.t(q[40])
    qc.t(q[41])
    qc.cz(q[42], q[43])
    qc.t(q[44])
    qc.t(q[45])
    qc.cz(q[46], q[47])
    qc.t(q[48])
    qc.ry(np.pi / 2, q[0])
    qc.ry(np.pi / 2, q[1])
    qc.rx(np.pi / 2, q[4])
    qc.ry(np.pi / 2, q[5])
    qc.cz(q[7], q[14])
    qc.cz(q[22], q[29])
    qc.cz(q[35], q[42])
    qc.t(q[0])
    qc.cz(q[1], q[2])
    qc.t(q[4])
    qc.cz(q[5], q[6])
    qc.ry(np.pi / 2, q[7])
    qc.cz(q[9], q[16])
    qc.rx(np.pi / 2, q[22])
    qc.ry(np.pi / 2, q[23])
    qc.cz(q[24], q[31])
    qc.rx(np.pi / 2, q[35])
    qc.cz(q[37], q[44])
    qc.cz(q[0], q[7])
    qc.ry(np.pi / 2, q[9])
    qc.ry(np.pi / 2, q[10])
    qc.cz(q[11], q[18])
    qc.t(q[23])
    qc.cz(q[24], q[25])
    qc.cz(q[26], q[33])
    qc.ry(np.pi / 2, q[37])
    qc.rx(np.pi / 2, q[38])
    qc.cz(q[39], q[46])
    qc.rx(np.pi / 2, q[0])
    qc.rx(np.pi / 2, q[1])
    qc.cz(q[2], q[9])
    qc.cz(q[10], q[11])
    qc.cz(q[13], q[20])
    qc.rx(np.pi / 2, q[26])
    qc.ry(np.pi / 2, q[28])
    qc.cz(q[29], q[30])
    qc.ry(np.pi / 2, q[31])
    qc.rx(np.pi / 2, q[32])
    qc.cz(q[33], q[34])
    qc.cz(q[38], q[39])
    qc.cz(q[41], q[48])
    qc.t(q[0])
    qc.t(q[1])
    qc.cz(q[2], q[3])
    qc.cz(q[4], q[11])
    qc.rx(np.pi / 2, q[13])
    qc.rx(np.pi / 2, q[14])
    qc.rx(np.pi / 2, q[15])
    qc.rx(np.pi / 2, q[18])
    qc.ry(np.pi / 2, q[19])
    qc.t(q[28])
    qc.t(q[32])
    qc.rx(np.pi / 2, q[41])
    qc.rx(np.pi / 2, q[42])
    qc.rx(np.pi / 2, q[43])
    qc.rx(np.pi / 2, q[46])
    qc.rx(np.pi / 2, q[47])
    qc.h(q[0])
    qc.ry(np.pi / 2, q[2])
    qc.rx(np.pi / 2, q[3])
    qc.ry(np.pi / 2, q[4])
    qc.rx(np.pi / 2, q[5])
    qc.cz(q[6], q[13])
    qc.t(q[14])
    qc.cz(q[15], q[16])
    qc.cz(q[19], q[20])
    qc.cz(q[28], q[35])
    qc.t(q[42])
    qc.cz(q[43], q[44])
    qc.t(q[46])
    qc.cz(q[47], q[48])
    qc.t(q[2])
    qc.t(q[4])
    qc.t(q[5])
    qc.rx(np.pi / 2, q[6])
    qc.cz(q[7], q[8])
    qc.ry(np.pi / 2, q[9])
    qc.ry(np.pi / 2, q[10])
    qc.cz(q[11], q[12])
    qc.ry(np.pi / 2, q[13])
    qc.cz(q[15], q[22])
    qc.rx(np.pi / 2, q[28])
    qc.rx(np.pi / 2, q[29])
    qc.cz(q[30], q[37])
    qc.rx(np.pi / 2, q[43])
    qc.rx(np.pi / 2, q[44])
    qc.rx(np.pi / 2, q[47])
    qc.ry(np.pi / 2, q[48])
    qc.cz(q[3], q[4])
    qc.t(q[6])
    qc.rx(np.pi / 2, q[7])
    qc.t(q[10])
    qc.rx(np.pi / 2, q[15])
    qc.rx(np.pi / 2, q[16])
    qc.cz(q[17], q[24])
    qc.t(q[29])
    qc.t(q[31])
    qc.cz(q[32], q[39])
    qc.t(q[43])
    qc.cz(q[44], q[45])
    qc.t(q[47])
    qc.t(q[48])
    qc.t(q[7])
    qc.cz(q[8], q[15])
    qc.cz(q[16], q[17])
    qc.t(q[18])
    qc.cz(q[19], q[26])
    qc.cz(q[30], q[31])
    qc.rx(np.pi / 2, q[32])
    qc.rx(np.pi / 2, q[33])
    qc.cz(q[34], q[41])
    qc.h(q[48])
    qc.t(q[9])
    qc.cz(q[10], q[17])
    qc.rx(np.pi / 2, q[19])
    qc.ry(np.pi / 2, q[20])
    qc.cz(q[21], q[22])
    qc.rx(np.pi / 2, q[24])
    qc.ry(np.pi / 2, q[25])
    qc.t(q[33])
    qc.ry(np.pi / 2, q[34])
    qc.cz(q[35], q[36])
    qc.ry(np.pi / 2, q[37])
    qc.rx(np.pi / 2, q[38])
    qc.cz(q[39], q[40])
    qc.rx(np.pi / 2, q[41])
    qc.cz(q[8], q[9])
    qc.ry(np.pi / 2, q[10])
    qc.rx(np.pi / 2, q[11])
    qc.cz(q[12], q[19])
    qc.t(q[20])
    qc.cz(q[25], q[26])
    qc.rx(np.pi / 2, q[35])
    qc.t(q[38])
    qc.cz(q[1], q[8])
    qc.t(q[11])
    qc.t(q[13])
    qc.rx(np.pi / 2, q[15])
    qc.ry(np.pi / 2, q[16])
    qc.cz(q[17], q[18])
    qc.rx(np.pi / 2, q[19])
    qc.cz(q[21], q[28])
    qc.t(q[35])
    qc.cz(q[36], q[43])
    qc.h(q[1])
    qc.h(q[2])
    qc.cz(q[3], q[10])
    qc.cz(q[12], q[13])
    qc.t(q[16])
    qc.ry(np.pi / 2, q[21])
    qc.rx(np.pi / 2, q[22])
    qc.cz(q[23], q[30])
    qc.t(q[37])
    qc.cz(q[38], q[45])
    qc.h(q[3])
    qc.ry(np.pi / 2, q[4])
    qc.cz(q[5], q[12])
    qc.ry(np.pi / 2, q[13])
    qc.cz(q[14], q[21])
    qc.cz(q[22], q[23])
    qc.t(q[24])
    qc.cz(q[25], q[32])
    qc.cz(q[36], q[37])
    qc.ry(np.pi / 2, q[38])
    qc.rx(np.pi / 2, q[39])
    qc.cz(q[40], q[47])
    qc.h(q[4])
    qc.h(q[5])
    qc.h(q[6])
    qc.h(q[7])
    qc.h(q[8])
    qc.ry(np.pi / 2, q[9])
    qc.h(q[10])
    qc.h(q[11])
    qc.h(q[12])
    qc.h(q[13])
    qc.h(q[14])
    qc.t(q[15])
    qc.cz(q[16], q[23])
    qc.rx(np.pi / 2, q[25])
    qc.rx(np.pi / 2, q[26])
    qc.cz(q[27], q[34])
    qc.t(q[39])
    qc.t(q[41])
    qc.h(q[42])
    qc.rx(np.pi / 2, q[43])
    qc.ry(np.pi / 2, q[44])
    qc.cz(q[45], q[46])
    qc.ry(np.pi / 2, q[47])
    qc.h(q[9])
    qc.h(q[15])
    qc.h(q[16])
    qc.ry(np.pi / 2, q[17])
    qc.cz(q[18], q[25])
    qc.cz(q[26], q[27])
    qc.ry(np.pi / 2, q[28])
    qc.rx(np.pi / 2, q[30])
    qc.rx(np.pi / 2, q[31])
    qc.rx(np.pi / 2, q[34])
    qc.cz(q[40], q[41])
    qc.t(q[43])
    qc.t(q[44])
    qc.rx(np.pi / 2, q[45])
    qc.rx(np.pi / 2, q[46])
    qc.t(q[47])
    qc.h(q[17])
    qc.h(q[18])
    qc.t(q[19])
    qc.cz(q[20], q[27])
    qc.t(q[28])
    qc.cz(q[31], q[32])
    qc.ry(np.pi / 2, q[41])
    qc.h(q[43])
    qc.h(q[44])
    qc.h(q[45])
    qc.h(q[46])
    qc.h(q[47])
    qc.h(q[19])
    qc.h(q[20])
    qc.h(q[21])
    qc.rx(np.pi / 2, q[22])
    qc.h(q[23])
    qc.h(q[24])
    qc.h(q[25])
    qc.ry(np.pi / 2, q[26])
    qc.h(q[27])
    qc.h(q[28])
    qc.cz(q[29], q[36])
    qc.h(q[41])
    qc.h(q[22])
    qc.h(q[26])
    qc.h(q[29])
    qc.t(q[30])
    qc.cz(q[31], q[38])
    qc.h(q[30])
    qc.h(q[31])
    qc.ry(np.pi / 2, q[32])
    qc.cz(q[33], q[40])
    qc.h(q[32])
    qc.h(q[33])
    qc.t(q[34])
    qc.h(q[35])
    qc.h(q[36])
    qc.ry(np.pi / 2, q[37])
    qc.h(q[38])
    qc.h(q[39])
    qc.h(q[40])
    qc.h(q[34])
    qc.h(q[37])
    return qc

def supremacy_35():
    qc = QuantumCircuit()

    q = QuantumRegister(35, 'q')

    qc.add_register(q)

    qc.h(q[0])
    qc.h(q[1])
    qc.h(q[2])
    qc.h(q[3])
    qc.h(q[4])
    qc.h(q[5])
    qc.h(q[6])
    qc.h(q[7])
    qc.h(q[8])
    qc.h(q[9])
    qc.h(q[10])
    qc.h(q[11])
    qc.h(q[12])
    qc.h(q[13])
    qc.h(q[14])
    qc.h(q[15])
    qc.h(q[16])
    qc.h(q[17])
    qc.h(q[18])
    qc.h(q[19])
    qc.h(q[20])
    qc.h(q[21])
    qc.h(q[22])
    qc.h(q[23])
    qc.h(q[24])
    qc.h(q[25])
    qc.h(q[26])
    qc.h(q[27])
    qc.h(q[28])
    qc.h(q[29])
    qc.h(q[30])
    qc.h(q[31])
    qc.h(q[32])
    qc.h(q[33])
    qc.h(q[34])
    qc.cz(q[0], q[1])
    qc.t(q[2])
    qc.t(q[3])
    qc.cz(q[4], q[5])
    qc.t(q[6])
    qc.t(q[7])
    qc.t(q[8])
    qc.cz(q[9], q[10])
    qc.t(q[11])
    qc.t(q[12])
    qc.t(q[13])
    qc.cz(q[14], q[15])
    qc.t(q[16])
    qc.t(q[17])
    qc.cz(q[18], q[19])
    qc.t(q[20])
    qc.t(q[21])
    qc.t(q[22])
    qc.cz(q[23], q[24])
    qc.t(q[25])
    qc.t(q[26])
    qc.t(q[27])
    qc.cz(q[28], q[29])
    qc.t(q[30])
    qc.t(q[31])
    qc.cz(q[32], q[33])
    qc.t(q[34])
    qc.ry(np.pi / 2, q[0])
    qc.rx(np.pi / 2, q[1])
    qc.ry(np.pi / 2, q[4])
    qc.ry(np.pi / 2, q[5])
    qc.cz(q[7], q[14])
    qc.cz(q[22], q[29])
    qc.t(q[0])
    qc.cz(q[1], q[2])
    qc.t(q[4])
    qc.cz(q[5], q[6])
    qc.ry(np.pi / 2, q[7])
    qc.cz(q[9], q[16])
    qc.ry(np.pi / 2, q[22])
    qc.ry(np.pi / 2, q[23])
    qc.cz(q[24], q[31])
    qc.cz(q[0], q[7])
    qc.rx(np.pi / 2, q[9])
    qc.ry(np.pi / 2, q[10])
    qc.cz(q[11], q[18])
    qc.t(q[23])
    qc.cz(q[24], q[25])
    qc.cz(q[26], q[33])
    qc.rx(np.pi / 2, q[0])
    qc.ry(np.pi / 2, q[1])
    qc.cz(q[2], q[9])
    qc.cz(q[10], q[11])
    qc.cz(q[13], q[20])
    qc.ry(np.pi / 2, q[26])
    qc.rx(np.pi / 2, q[28])
    qc.cz(q[29], q[30])
    qc.rx(np.pi / 2, q[31])
    qc.rx(np.pi / 2, q[32])
    qc.cz(q[33], q[34])
    qc.t(q[0])
    qc.t(q[1])
    qc.cz(q[2], q[3])
    qc.cz(q[4], q[11])
    qc.rx(np.pi / 2, q[13])
    qc.rx(np.pi / 2, q[14])
    qc.rx(np.pi / 2, q[15])
    qc.rx(np.pi / 2, q[18])
    qc.rx(np.pi / 2, q[19])
    qc.t(q[28])
    qc.rx(np.pi / 2, q[29])
    qc.rx(np.pi / 2, q[30])
    qc.t(q[31])
    qc.t(q[32])
    qc.ry(np.pi / 2, q[33])
    qc.rx(np.pi / 2, q[34])
    qc.h(q[0])
    qc.rx(np.pi / 2, q[2])
    qc.rx(np.pi / 2, q[3])
    qc.rx(np.pi / 2, q[4])
    qc.ry(np.pi / 2, q[5])
    qc.cz(q[6], q[13])
    qc.t(q[14])
    qc.cz(q[15], q[16])
    qc.cz(q[19], q[20])
    qc.t(q[29])
    qc.cz(q[30], q[31])
    qc.t(q[33])
    qc.t(q[34])
    qc.t(q[2])
    qc.t(q[4])
    qc.t(q[5])
    qc.rx(np.pi / 2, q[6])
    qc.cz(q[7], q[8])
    qc.ry(np.pi / 2, q[9])
    qc.rx(np.pi / 2, q[10])
    qc.cz(q[11], q[12])
    qc.ry(np.pi / 2, q[13])
    qc.cz(q[15], q[22])
    qc.cz(q[3], q[4])
    qc.t(q[6])
    qc.rx(np.pi / 2, q[7])
    qc.t(q[10])
    qc.rx(np.pi / 2, q[15])
    qc.ry(np.pi / 2, q[16])
    qc.cz(q[17], q[24])
    qc.t(q[7])
    qc.cz(q[8], q[15])
    qc.cz(q[16], q[17])
    qc.t(q[18])
    qc.cz(q[19], q[26])
    qc.t(q[9])
    qc.cz(q[10], q[17])
    qc.ry(np.pi / 2, q[19])
    qc.rx(np.pi / 2, q[20])
    qc.cz(q[21], q[22])
    qc.ry(np.pi / 2, q[24])
    qc.ry(np.pi / 2, q[25])
    qc.cz(q[8], q[9])
    qc.ry(np.pi / 2, q[10])
    qc.rx(np.pi / 2, q[11])
    qc.cz(q[12], q[19])
    qc.t(q[20])
    qc.cz(q[25], q[26])
    qc.cz(q[1], q[8])
    qc.t(q[11])
    qc.t(q[13])
    qc.ry(np.pi / 2, q[15])
    qc.ry(np.pi / 2, q[16])
    qc.cz(q[17], q[18])
    qc.ry(np.pi / 2, q[19])
    qc.cz(q[21], q[28])
    qc.h(q[1])
    qc.h(q[2])
    qc.cz(q[3], q[10])
    qc.cz(q[12], q[13])
    qc.t(q[16])
    qc.rx(np.pi / 2, q[21])
    qc.ry(np.pi / 2, q[22])
    qc.cz(q[23], q[30])
    qc.h(q[3])
    qc.ry(np.pi / 2, q[4])
    qc.cz(q[5], q[12])
    qc.ry(np.pi / 2, q[13])
    qc.cz(q[14], q[21])
    qc.cz(q[22], q[23])
    qc.t(q[24])
    qc.cz(q[25], q[32])
    qc.h(q[4])
    qc.h(q[5])
    qc.h(q[6])
    qc.h(q[7])
    qc.h(q[8])
    qc.ry(np.pi / 2, q[9])
    qc.h(q[10])
    qc.h(q[11])
    qc.h(q[12])
    qc.h(q[13])
    qc.h(q[14])
    qc.t(q[15])
    qc.cz(q[16], q[23])
    qc.rx(np.pi / 2, q[25])
    qc.ry(np.pi / 2, q[26])
    qc.cz(q[27], q[34])
    qc.h(q[9])
    qc.h(q[15])
    qc.h(q[16])
    qc.rx(np.pi / 2, q[17])
    qc.cz(q[18], q[25])
    qc.cz(q[26], q[27])
    qc.rx(np.pi / 2, q[28])
    qc.h(q[29])
    qc.ry(np.pi / 2, q[30])
    qc.rx(np.pi / 2, q[31])
    qc.h(q[33])
    qc.rx(np.pi / 2, q[34])
    qc.h(q[17])
    qc.h(q[18])
    qc.t(q[19])
    qc.cz(q[20], q[27])
    qc.t(q[28])
    qc.t(q[30])
    qc.cz(q[31], q[32])
    qc.t(q[34])
    qc.h(q[19])
    qc.h(q[20])
    qc.h(q[21])
    qc.rx(np.pi / 2, q[22])
    qc.h(q[23])
    qc.h(q[24])
    qc.h(q[25])
    qc.ry(np.pi / 2, q[26])
    qc.h(q[27])
    qc.h(q[28])
    qc.h(q[30])
    qc.rx(np.pi / 2, q[31])
    qc.ry(np.pi / 2, q[32])
    qc.h(q[34])
    qc.h(q[22])
    qc.h(q[26])
    qc.h(q[31])
    qc.h(q[32])
    return qc

