OPENQASM 2.0;
include "qelib1.inc";
qreg q[18];
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
rz(-0.7437681946593673) q[4];
rz(-0.7437681946593673) q[6];
cx q[4],q[6];
rz(0.7437681946593673) q[6];
cx q[4],q[6];
rz(-0.7437681946593673) q[4];
rz(-0.7437681946593673) q[8];
cx q[4],q[8];
rz(0.7437681946593673) q[8];
cx q[4],q[8];
rz(-0.7437681946593673) q[4];
rz(-0.7437681946593673) q[13];
cx q[4],q[13];
rz(0.7437681946593673) q[13];
cx q[4],q[13];
rz(-0.7437681946593673) q[6];
rz(-0.7437681946593673) q[3];
cx q[6],q[3];
rz(0.7437681946593673) q[3];
cx q[6],q[3];
rz(-0.7437681946593673) q[6];
rz(-0.7437681946593673) q[7];
cx q[6],q[7];
rz(0.7437681946593673) q[7];
cx q[6],q[7];
rz(-0.7437681946593673) q[3];
rz(-0.7437681946593673) q[16];
cx q[3],q[16];
rz(0.7437681946593673) q[16];
cx q[3],q[16];
rz(-0.7437681946593673) q[3];
rz(-0.7437681946593673) q[9];
cx q[3],q[9];
rz(0.7437681946593673) q[9];
cx q[3],q[9];
rz(-0.7437681946593673) q[16];
rz(-0.7437681946593673) q[15];
cx q[16],q[15];
rz(0.7437681946593673) q[15];
cx q[16],q[15];
rz(-0.7437681946593673) q[16];
rz(-0.7437681946593673) q[1];
cx q[16],q[1];
rz(0.7437681946593673) q[1];
cx q[16],q[1];
rz(-0.7437681946593673) q[5];
rz(-0.7437681946593673) q[10];
cx q[5],q[10];
rz(0.7437681946593673) q[10];
cx q[5],q[10];
rz(-0.7437681946593673) q[5];
rz(-0.7437681946593673) q[1];
cx q[5],q[1];
rz(0.7437681946593673) q[1];
cx q[5],q[1];
rz(-0.7437681946593673) q[5];
rz(-0.7437681946593673) q[11];
cx q[5],q[11];
rz(0.7437681946593673) q[11];
cx q[5],q[11];
rz(-0.7437681946593673) q[10];
rz(-0.7437681946593673) q[12];
cx q[10],q[12];
rz(0.7437681946593673) q[12];
cx q[10],q[12];
rz(-0.7437681946593673) q[10];
rz(-0.7437681946593673) q[0];
cx q[10],q[0];
rz(0.7437681946593673) q[0];
cx q[10],q[0];
rz(-0.7437681946593673) q[9];
rz(-0.7437681946593673) q[14];
cx q[9],q[14];
rz(0.7437681946593673) q[14];
cx q[9],q[14];
rz(-0.7437681946593673) q[9];
rz(-0.7437681946593673) q[11];
cx q[9],q[11];
rz(0.7437681946593673) q[11];
cx q[9],q[11];
rz(-0.7437681946593673) q[14];
rz(-0.7437681946593673) q[15];
cx q[14],q[15];
rz(0.7437681946593673) q[15];
cx q[14],q[15];
rz(-0.7437681946593673) q[14];
rz(-0.7437681946593673) q[1];
cx q[14],q[1];
rz(0.7437681946593673) q[1];
cx q[14],q[1];
rz(-0.7437681946593673) q[8];
rz(-0.7437681946593673) q[12];
cx q[8],q[12];
rz(0.7437681946593673) q[12];
cx q[8],q[12];
rz(-0.7437681946593673) q[8];
rz(-0.7437681946593673) q[2];
cx q[8],q[2];
rz(0.7437681946593673) q[2];
cx q[8],q[2];
rz(-0.7437681946593673) q[12];
rz(-0.7437681946593673) q[2];
cx q[12],q[2];
rz(0.7437681946593673) q[2];
cx q[12],q[2];
rz(-0.7437681946593673) q[11];
rz(-0.7437681946593673) q[17];
cx q[11],q[17];
rz(0.7437681946593673) q[17];
cx q[11],q[17];
rz(-0.7437681946593673) q[0];
rz(-0.7437681946593673) q[17];
cx q[0],q[17];
rz(0.7437681946593673) q[17];
cx q[0],q[17];
rz(-0.7437681946593673) q[0];
rz(-0.7437681946593673) q[15];
cx q[0],q[15];
rz(0.7437681946593673) q[15];
cx q[0],q[15];
rz(-0.7437681946593673) q[17];
rz(-0.7437681946593673) q[13];
cx q[17],q[13];
rz(0.7437681946593673) q[13];
cx q[17],q[13];
rz(-0.7437681946593673) q[2];
rz(-0.7437681946593673) q[7];
cx q[2],q[7];
rz(0.7437681946593673) q[7];
cx q[2],q[7];
rz(-0.7437681946593673) q[13];
rz(-0.7437681946593673) q[7];
cx q[13],q[7];
rz(0.7437681946593673) q[7];
cx q[13],q[7];
rx(-0.23086541119537607) q[0];
rx(-0.23086541119537607) q[1];
rx(-0.23086541119537607) q[2];
rx(-0.23086541119537607) q[3];
rx(-0.23086541119537607) q[4];
rx(-0.23086541119537607) q[5];
rx(-0.23086541119537607) q[6];
rx(-0.23086541119537607) q[7];
rx(-0.23086541119537607) q[8];
rx(-0.23086541119537607) q[9];
rx(-0.23086541119537607) q[10];
rx(-0.23086541119537607) q[11];
rx(-0.23086541119537607) q[12];
rx(-0.23086541119537607) q[13];
rx(-0.23086541119537607) q[14];
rx(-0.23086541119537607) q[15];
rx(-0.23086541119537607) q[16];
rx(-0.23086541119537607) q[17];