OPENQASM 2.0;
include "qelib1.inc";
qreg q[15];
h q[0];
rz(-0.49859525557125184) q[0];
h q[1];
rz(-0.49859525557125184) q[1];
h q[2];
rz(-0.49859525557125184) q[2];
cx q[0],q[2];
rz(0.49859525557125184) q[2];
cx q[0],q[2];
rz(-0.49859525557125184) q[0];
rz(-0.49859525557125184) q[2];
h q[3];
rz(-0.49859525557125184) q[3];
h q[4];
rz(-0.49859525557125184) q[4];
cx q[3],q[4];
rz(0.49859525557125184) q[4];
cx q[3],q[4];
rz(-0.49859525557125184) q[3];
rz(-0.49859525557125184) q[4];
h q[5];
rz(-0.49859525557125184) q[5];
cx q[4],q[5];
rz(0.49859525557125184) q[5];
cx q[4],q[5];
rz(-0.49859525557125184) q[4];
rx(-3.848540463051039) q[5];
h q[6];
rz(-0.49859525557125184) q[6];
cx q[0],q[6];
rz(0.49859525557125184) q[6];
cx q[0],q[6];
rz(-0.49859525557125184) q[0];
rz(-0.49859525557125184) q[6];
h q[8];
rz(-0.49859525557125184) q[8];
cx q[0],q[8];
rz(0.49859525557125184) q[8];
cx q[0],q[8];
rx(-3.848540463051039) q[0];
rz(-0.49859525557125184) q[8];
h q[9];
rz(-0.49859525557125184) q[9];
cx q[1],q[9];
rz(0.49859525557125184) q[9];
cx q[1],q[9];
rz(-0.49859525557125184) q[9];
rx(-3.848540463051039) q[1];
cx q[6],q[9];
rz(0.49859525557125184) q[9];
cx q[6],q[9];
rz(-0.49859525557125184) q[9];
rz(-0.49859525557125184) q[6];
h q[10];
rz(-0.49859525557125184) q[10];
cx q[4],q[10];
rz(0.49859525557125184) q[10];
cx q[4],q[10];
rz(-0.49859525557125184) q[10];
rx(-3.848540463051039) q[4];
rz(-0.49859525557125184) q[7];
rz(-0.49859525557125184) q[11];
h q[12];
rz(-0.49859525557125184) q[12];
cx q[6],q[12];
rz(0.49859525557125184) q[12];
cx q[6],q[12];
rz(-0.49859525557125184) q[12];
rz(-0.49859525557125184) q[6];
cx q[7],q[12];
rz(0.49859525557125184) q[12];
cx q[7],q[12];
rz(-0.49859525557125184) q[12];
cx q[10],q[12];
rz(0.49859525557125184) q[12];
cx q[10],q[12];
rx(-3.848540463051039) q[10];
rx(-3.848540463051039) q[12];
rx(-3.848540463051039) q[7];
h q[13];
rz(-0.49859525557125184) q[13];
cx q[11],q[13];
rz(0.49859525557125184) q[13];
cx q[11],q[13];
rx(-3.848540463051039) q[11];
rx(-3.848540463051039) q[13];
h q[14];
rz(-0.49859525557125184) q[14];
cx q[6],q[14];
rz(0.49859525557125184) q[14];
cx q[6],q[14];
rz(-0.49859525557125184) q[14];
rx(-3.848540463051039) q[6];