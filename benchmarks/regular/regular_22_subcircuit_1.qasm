OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
h q[0];
rz(0.03406927200330334) q[0];
h q[1];
rz(0.03406927200330334) q[1];
h q[2];
rz(0.03406927200330334) q[2];
h q[4];
rz(0.03406927200330334) q[4];
h q[5];
rz(0.03406927200330334) q[5];
cx q[6],q[0];
rz(-0.03406927200330334) q[0];
cx q[6],q[0];
rz(0.03406927200330334) q[0];
rx(2.288070191853295) q[6];
cx q[3],q[5];
rz(-0.03406927200330334) q[5];
cx q[3],q[5];
rx(2.288070191853295) q[3];
rz(0.03406927200330334) q[5];
cx q[8],q[2];
rz(-0.03406927200330334) q[2];
cx q[8],q[2];
rz(0.03406927200330334) q[8];
cx q[8],q[4];
rz(-0.03406927200330334) q[4];
cx q[8],q[4];
rz(0.03406927200330334) q[4];
cx q[5],q[4];
rz(-0.03406927200330334) q[4];
cx q[5],q[4];
rz(0.03406927200330334) q[4];
rz(0.03406927200330334) q[5];
cx q[5],q[1];
rz(-0.03406927200330334) q[1];
cx q[5],q[1];
rz(0.03406927200330334) q[1];
rx(2.288070191853295) q[5];
rx(2.288070191853295) q[8];
rz(0.03406927200330334) q[2];
cx q[1],q[2];
rz(-0.03406927200330334) q[2];
cx q[1],q[2];
rz(0.03406927200330334) q[1];
cx q[1],q[0];
rz(-0.03406927200330334) q[0];
cx q[1],q[0];
rz(0.03406927200330334) q[0];
rx(2.288070191853295) q[1];
cx q[7],q[0];
rz(-0.03406927200330334) q[0];
cx q[7],q[0];
rx(2.288070191853295) q[0];
rx(2.288070191853295) q[7];
rz(0.03406927200330334) q[2];
cx q[2],q[4];
rz(-0.03406927200330334) q[4];
cx q[2],q[4];
rx(2.288070191853295) q[4];
rx(2.288070191853295) q[2];