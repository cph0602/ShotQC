OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
h q[8];
h q[9];
h q[10];
h q[11];
h q[12];
h q[13];
h q[14];
h q[15];
h q[16];
h q[17];
h q[18];
h q[19];
h q[20];
h q[21];
h q[22];
h q[23];
h q[24];
h q[25];
h q[26];
rz(1.173389172278179) q[0];
rz(1.173389172278179) q[12];
cx q[0],q[12];
rz(-1.173389172278179) q[12];
cx q[0],q[12];
rz(1.173389172278179) q[0];
rz(1.173389172278179) q[17];
cx q[0],q[17];
rz(-1.173389172278179) q[17];
cx q[0],q[17];
rz(1.173389172278179) q[1];
rz(1.173389172278179) q[12];
cx q[1],q[12];
rz(-1.173389172278179) q[12];
cx q[1],q[12];
rz(1.173389172278179) q[1];
rz(1.173389172278179) q[18];
cx q[1],q[18];
rz(-1.173389172278179) q[18];
cx q[1],q[18];
rz(1.173389172278179) q[2];
rz(1.173389172278179) q[4];
cx q[2],q[4];
rz(-1.173389172278179) q[4];
cx q[2],q[4];
rz(1.173389172278179) q[2];
rz(1.173389172278179) q[11];
cx q[2],q[11];
rz(-1.173389172278179) q[11];
cx q[2],q[11];
rz(1.173389172278179) q[3];
rz(1.173389172278179) q[7];
cx q[3],q[7];
rz(-1.173389172278179) q[7];
cx q[3],q[7];
rz(1.173389172278179) q[3];
rz(1.173389172278179) q[14];
cx q[3],q[14];
rz(-1.173389172278179) q[14];
cx q[3],q[14];
rz(1.173389172278179) q[3];
rz(1.173389172278179) q[18];
cx q[3],q[18];
rz(-1.173389172278179) q[18];
cx q[3],q[18];
rz(1.173389172278179) q[4];
rz(1.173389172278179) q[13];
cx q[4],q[13];
rz(-1.173389172278179) q[13];
cx q[4],q[13];
rz(1.173389172278179) q[4];
rz(1.173389172278179) q[18];
cx q[4],q[18];
rz(-1.173389172278179) q[18];
cx q[4],q[18];
rz(1.173389172278179) q[5];
rz(1.173389172278179) q[12];
cx q[5],q[12];
rz(-1.173389172278179) q[12];
cx q[5],q[12];
rz(1.173389172278179) q[5];
rz(1.173389172278179) q[13];
cx q[5],q[13];
rz(-1.173389172278179) q[13];
cx q[5],q[13];
rz(1.173389172278179) q[5];
rz(1.173389172278179) q[15];
cx q[5],q[15];
rz(-1.173389172278179) q[15];
cx q[5],q[15];
rz(1.173389172278179) q[5];
rz(1.173389172278179) q[20];
cx q[5],q[20];
rz(-1.173389172278179) q[20];
cx q[5],q[20];
rz(1.173389172278179) q[6];
rz(1.173389172278179) q[26];
cx q[6],q[26];
rz(-1.173389172278179) q[26];
cx q[6],q[26];
rz(1.173389172278179) q[7];
rz(1.173389172278179) q[19];
cx q[7],q[19];
rz(-1.173389172278179) q[19];
cx q[7],q[19];
rz(1.173389172278179) q[8];
rz(1.173389172278179) q[11];
cx q[8],q[11];
rz(-1.173389172278179) q[11];
cx q[8],q[11];
rz(1.173389172278179) q[8];
rz(1.173389172278179) q[20];
cx q[8],q[20];
rz(-1.173389172278179) q[20];
cx q[8],q[20];
rz(1.173389172278179) q[8];
rz(1.173389172278179) q[26];
cx q[8],q[26];
rz(-1.173389172278179) q[26];
cx q[8],q[26];
rz(1.173389172278179) q[9];
rz(1.173389172278179) q[16];
cx q[9],q[16];
rz(-1.173389172278179) q[16];
cx q[9],q[16];
rz(1.173389172278179) q[9];
rz(1.173389172278179) q[20];
cx q[9],q[20];
rz(-1.173389172278179) q[20];
cx q[9],q[20];
rz(1.173389172278179) q[9];
rz(1.173389172278179) q[22];
cx q[9],q[22];
rz(-1.173389172278179) q[22];
cx q[9],q[22];
rz(1.173389172278179) q[10];
rz(1.173389172278179) q[12];
cx q[10],q[12];
rz(-1.173389172278179) q[12];
cx q[10],q[12];
rz(1.173389172278179) q[10];
rz(1.173389172278179) q[14];
cx q[10],q[14];
rz(-1.173389172278179) q[14];
cx q[10],q[14];
rz(1.173389172278179) q[11];
rz(1.173389172278179) q[19];
cx q[11],q[19];
rz(-1.173389172278179) q[19];
cx q[11],q[19];
rz(1.173389172278179) q[12];
rz(1.173389172278179) q[14];
cx q[12],q[14];
rz(-1.173389172278179) q[14];
cx q[12],q[14];
rz(1.173389172278179) q[14];
rz(1.173389172278179) q[21];
cx q[14],q[21];
rz(-1.173389172278179) q[21];
cx q[14],q[21];
rz(1.173389172278179) q[14];
rz(1.173389172278179) q[23];
cx q[14],q[23];
rz(-1.173389172278179) q[23];
cx q[14],q[23];
rz(1.173389172278179) q[14];
rz(1.173389172278179) q[25];
cx q[14],q[25];
rz(-1.173389172278179) q[25];
cx q[14],q[25];
rz(1.173389172278179) q[16];
rz(1.173389172278179) q[22];
cx q[16],q[22];
rz(-1.173389172278179) q[22];
cx q[16],q[22];
rz(1.173389172278179) q[17];
rz(1.173389172278179) q[24];
cx q[17],q[24];
rz(-1.173389172278179) q[24];
cx q[17],q[24];
rz(1.173389172278179) q[20];
rz(1.173389172278179) q[21];
cx q[20],q[21];
rz(-1.173389172278179) q[21];
cx q[20],q[21];
rz(1.173389172278179) q[20];
rz(1.173389172278179) q[22];
cx q[20],q[22];
rz(-1.173389172278179) q[22];
cx q[20],q[22];
rz(1.173389172278179) q[21];
rz(1.173389172278179) q[23];
cx q[21],q[23];
rz(-1.173389172278179) q[23];
cx q[21],q[23];
rz(1.173389172278179) q[23];
rz(1.173389172278179) q[24];
cx q[23],q[24];
rz(-1.173389172278179) q[24];
cx q[23],q[24];
rx(2.53398799707007) q[0];
rx(2.53398799707007) q[1];
rx(2.53398799707007) q[2];
rx(2.53398799707007) q[3];
rx(2.53398799707007) q[4];
rx(2.53398799707007) q[5];
rx(2.53398799707007) q[6];
rx(2.53398799707007) q[7];
rx(2.53398799707007) q[8];
rx(2.53398799707007) q[9];
rx(2.53398799707007) q[10];
rx(2.53398799707007) q[11];
rx(2.53398799707007) q[12];
rx(2.53398799707007) q[13];
rx(2.53398799707007) q[14];
rx(2.53398799707007) q[15];
rx(2.53398799707007) q[16];
rx(2.53398799707007) q[17];
rx(2.53398799707007) q[18];
rx(2.53398799707007) q[19];
rx(2.53398799707007) q[20];
rx(2.53398799707007) q[21];
rx(2.53398799707007) q[22];
rx(2.53398799707007) q[23];
rx(2.53398799707007) q[24];
rx(2.53398799707007) q[25];
rx(2.53398799707007) q[26];