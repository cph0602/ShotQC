OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
rz(-1.4236524968203819) q[0];
rz(-1.4236524968203819) q[6];
cx q[0],q[6];
rz(1.4236524968203819) q[6];
cx q[0],q[6];
rz(-1.4236524968203819) q[1];
rz(-1.4236524968203819) q[4];
cx q[1],q[4];
rz(1.4236524968203819) q[4];
cx q[1],q[4];
rz(-1.4236524968203819) q[1];
rz(-1.4236524968203819) q[6];
cx q[1],q[6];
rz(1.4236524968203819) q[6];
cx q[1],q[6];
rz(-1.4236524968203819) q[2];
rz(-1.4236524968203819) q[5];
cx q[2],q[5];
rz(1.4236524968203819) q[5];
cx q[2],q[5];
rz(-1.4236524968203819) q[3];
rz(-1.4236524968203819) q[5];
cx q[3],q[5];
rz(1.4236524968203819) q[5];
cx q[3],q[5];
rz(-1.4236524968203819) q[5];
rz(-1.4236524968203819) q[6];
cx q[5],q[6];
rz(1.4236524968203819) q[6];
cx q[5],q[6];
rx(-2.2768617554878254) q[0];
rx(-2.2768617554878254) q[1];
rx(-2.2768617554878254) q[2];
rx(-2.2768617554878254) q[3];
rx(-2.2768617554878254) q[4];
rx(-2.2768617554878254) q[5];
rx(-2.2768617554878254) q[6];