OPENQASM 2.0;
include "qelib1.inc";
qreg q[14];
h q[0];
rz(-1.0543470241078667) q[0];
h q[1];
rz(-1.0543470241078667) q[1];
h q[2];
rz(-1.0543470241078667) q[2];
cx q[0],q[2];
rz(1.0543470241078667) q[2];
cx q[0],q[2];
rz(-1.0543470241078667) q[0];
rz(-1.0543470241078667) q[2];
h q[3];
rz(-1.0543470241078667) q[3];
h q[4];
rz(-1.0543470241078667) q[4];
cx q[0],q[4];
rz(1.0543470241078667) q[4];
cx q[0],q[4];
rz(-1.0543470241078667) q[4];
rz(-1.0543470241078667) q[0];
h q[5];
rz(-1.0543470241078667) q[5];
h q[6];
rz(-1.0543470241078667) q[6];
h q[7];
rz(-1.0543470241078667) q[7];
h q[8];
rz(-1.0543470241078667) q[8];
h q[9];
rz(-1.0543470241078667) q[9];
h q[10];
rz(-1.0543470241078667) q[10];
cx q[2],q[10];
rz(1.0543470241078667) q[10];
cx q[2],q[10];
rz(-1.0543470241078667) q[10];
rz(-1.0543470241078667) q[2];
cx q[2],q[9];
rz(1.0543470241078667) q[9];
cx q[2],q[9];
rx(-0.2598741101971118) q[2];
h q[11];
rz(-1.0543470241078667) q[11];
h q[12];
rz(-1.0543470241078667) q[12];
h q[13];
rz(-1.0543470241078667) q[13];
cx q[7],q[13];
rz(1.0543470241078667) q[13];
cx q[7],q[13];
rz(-1.0543470241078667) q[7];
cx q[7],q[12];
rz(1.0543470241078667) q[12];
cx q[7],q[12];
rz(-1.0543470241078667) q[7];
cx q[7],q[3];
rz(-1.0543470241078667) q[12];
rz(-1.0543470241078667) q[13];
cx q[13],q[1];
cx q[0],q[12];
rz(1.0543470241078667) q[12];
cx q[0],q[12];
rz(-1.0543470241078667) q[12];
cx q[4],q[12];
rz(1.0543470241078667) q[12];
cx q[4],q[12];
rz(-1.0543470241078667) q[4];
rx(-0.2598741101971118) q[12];
rx(-0.2598741101971118) q[0];
rz(1.0543470241078667) q[1];
cx q[13],q[1];
rz(-1.0543470241078667) q[13];
cx q[13],q[5];
rz(1.0543470241078667) q[5];
cx q[13],q[5];
rx(-0.2598741101971118) q[13];
rz(-1.0543470241078667) q[1];
cx q[1],q[11];
rz(1.0543470241078667) q[11];
cx q[1],q[11];
rz(-1.0543470241078667) q[1];
cx q[1],q[6];
rz(1.0543470241078667) q[6];
cx q[1],q[6];
rz(-1.0543470241078667) q[6];
cx q[4],q[6];
rz(1.0543470241078667) q[6];
cx q[4],q[6];
rx(-0.2598741101971118) q[4];
rz(-1.0543470241078667) q[6];
cx q[10],q[6];
rz(1.0543470241078667) q[6];
cx q[10],q[6];
rx(-0.2598741101971118) q[6];
rz(-1.0543470241078667) q[10];
cx q[10],q[8];
rz(1.0543470241078667) q[8];
cx q[10],q[8];
rx(-0.2598741101971118) q[10];
rx(-0.2598741101971118) q[1];
rz(1.0543470241078667) q[3];
cx q[7],q[3];
rx(-0.2598741101971118) q[7];