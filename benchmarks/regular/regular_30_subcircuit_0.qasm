OPENQASM 2.0;
include "qelib1.inc";
qreg q[18];
h q[0];
rz(-0.39856068593594896) q[0];
h q[2];
rz(-0.39856068593594896) q[2];
h q[3];
rz(-0.39856068593594896) q[3];
h q[4];
rz(-0.39856068593594896) q[4];
h q[5];
rz(-0.39856068593594896) q[5];
h q[6];
rz(-0.39856068593594896) q[6];
h q[7];
rz(-0.39856068593594896) q[7];
cx q[3],q[7];
rz(0.39856068593594896) q[7];
cx q[3],q[7];
rz(-0.39856068593594896) q[7];
rz(-0.39856068593594896) q[3];
cx q[3],q[0];
rz(0.39856068593594896) q[0];
cx q[3],q[0];
rz(-0.39856068593594896) q[0];
cx q[0],q[4];
rz(-0.39856068593594896) q[3];
rz(0.39856068593594896) q[4];
cx q[0],q[4];
rz(-0.39856068593594896) q[0];
rz(-0.39856068593594896) q[4];
h q[8];
rz(-0.39856068593594896) q[8];
cx q[3],q[8];
rz(0.39856068593594896) q[8];
cx q[3],q[8];
rz(-0.39856068593594896) q[8];
rx(-4.081822163206743) q[3];
h q[9];
rz(-0.39856068593594896) q[9];
cx q[0],q[9];
rz(0.39856068593594896) q[9];
cx q[0],q[9];
rx(-4.081822163206743) q[0];
rz(-0.39856068593594896) q[9];
h q[10];
rz(-0.39856068593594896) q[10];
cx q[7],q[10];
rz(0.39856068593594896) q[10];
cx q[7],q[10];
rz(-0.39856068593594896) q[7];
rz(-0.39856068593594896) q[10];
h q[11];
rz(-0.39856068593594896) q[11];
h q[12];
rz(-0.39856068593594896) q[12];
h q[13];
rz(-0.39856068593594896) q[13];
cx q[11],q[13];
rz(0.39856068593594896) q[13];
cx q[11],q[13];
rz(-0.39856068593594896) q[11];
cx q[11],q[6];
rz(0.39856068593594896) q[6];
cx q[11],q[6];
rz(-0.39856068593594896) q[6];
rz(-0.39856068593594896) q[11];
rz(-0.39856068593594896) q[13];
h q[14];
rz(-0.39856068593594896) q[14];
cx q[7],q[14];
rz(0.39856068593594896) q[14];
cx q[7],q[14];
rx(-4.081822163206743) q[7];
rz(-0.39856068593594896) q[14];
h q[15];
rz(-0.39856068593594896) q[15];
cx q[11],q[15];
rz(0.39856068593594896) q[15];
cx q[11],q[15];
rx(-4.081822163206743) q[11];
rz(-0.39856068593594896) q[15];
cx q[9],q[15];
rz(0.39856068593594896) q[15];
cx q[9],q[15];
rz(-0.39856068593594896) q[9];
rz(-0.39856068593594896) q[15];
rz(-0.39856068593594896) q[1];
cx q[1],q[14];
rz(0.39856068593594896) q[14];
cx q[1],q[14];
rx(-4.081822163206743) q[1];
rz(-0.39856068593594896) q[14];
cx q[14],q[15];
rz(0.39856068593594896) q[15];
cx q[14],q[15];
rx(-4.081822163206743) q[14];
rx(-4.081822163206743) q[15];
h q[16];
rz(-0.39856068593594896) q[16];
cx q[13],q[16];
rz(0.39856068593594896) q[16];
cx q[13],q[16];
rz(-0.39856068593594896) q[13];
cx q[13],q[2];
rz(-0.39856068593594896) q[16];
rz(0.39856068593594896) q[2];
cx q[13],q[2];
rx(-4.081822163206743) q[13];
rz(-0.39856068593594896) q[2];
h q[17];
rz(-0.39856068593594896) q[17];
cx q[4],q[17];
rz(0.39856068593594896) q[17];
cx q[4],q[17];
rz(-0.39856068593594896) q[17];
cx q[17],q[5];
rz(-0.39856068593594896) q[4];
cx q[4],q[10];
rz(0.39856068593594896) q[10];
cx q[4],q[10];
rz(-0.39856068593594896) q[10];
cx q[10],q[6];
rz(0.39856068593594896) q[6];
cx q[10],q[6];
rz(-0.39856068593594896) q[6];
rx(-4.081822163206743) q[10];
rx(-4.081822163206743) q[4];
rz(0.39856068593594896) q[5];
cx q[17],q[5];
rz(-0.39856068593594896) q[17];
cx q[17],q[12];
rz(0.39856068593594896) q[12];
cx q[17],q[12];
rz(-0.39856068593594896) q[12];
cx q[12],q[16];
rz(0.39856068593594896) q[16];
cx q[12],q[16];
rz(-0.39856068593594896) q[12];
rz(-0.39856068593594896) q[16];
rx(-4.081822163206743) q[17];
rz(-0.39856068593594896) q[5];
cx q[12],q[5];
rz(0.39856068593594896) q[5];
cx q[12],q[5];
rx(-4.081822163206743) q[12];
rz(-0.39856068593594896) q[5];
cx q[5],q[2];
rz(0.39856068593594896) q[2];
cx q[5],q[2];
rz(-0.39856068593594896) q[2];
cx q[2],q[6];
rz(0.39856068593594896) q[6];
cx q[2],q[6];
rx(-4.081822163206743) q[6];
rx(-4.081822163206743) q[2];
rx(-4.081822163206743) q[5];