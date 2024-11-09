OPENQASM 2.0;
include "qelib1.inc";
qreg q[11];
h q[0];
h q[1];
cz q[0],q[1];
rx(pi/2) q[0];
t q[0];
ry(pi/2) q[1];
h q[2];
t q[2];
cz q[1],q[2];
ry(pi/2) q[1];
t q[1];
h q[3];
t q[3];
h q[4];
t q[4];
h q[6];
h q[7];
cz q[6],q[7];
ry(pi/2) q[7];
cz q[3],q[6];
ry(pi/2) q[6];
t q[6];
ry(pi/2) q[3];
cz q[0],q[3];
ry(pi/2) q[0];
t q[0];
h q[0];
cz q[3],q[4];
ry(pi/2) q[3];
t q[3];
h q[3];
h q[8];
t q[8];
cz q[5],q[8];
cz q[7],q[8];
rx(pi/2) q[8];
ry(pi/2) q[5];
cz q[2],q[5];
rx(pi/2) q[5];
t q[5];
h q[9];
t q[9];
h q[10];
t q[10];
cz q[7],q[10];
ry(pi/2) q[7];
cz q[9],q[10];
rx(pi/2) q[9];
t q[9];
cz q[6],q[9];
h q[6];
h q[9];
ry(pi/2) q[10];
cz q[4],q[7];
rx(pi/2) q[7];
t q[7];
h q[7];
cz q[4],q[5];
cz q[1],q[4];
h q[1];
h q[4];
ry(pi/2) q[5];
h q[5];