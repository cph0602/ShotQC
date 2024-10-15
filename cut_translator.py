
subcircuits = [['q[0]0 q[1]0', 'q[1]1 q[2]0', 'q[7]0 q[8]0', 'q[0]1 q[5]1', 'q[5]2 q[6]0', 'q[7]1 q[12]0', 'q[2]1 q[7]2', 'q[2]2 q[3]0', 'q[9]0 q[14]0', 'q[8]1 q[9]1', 'q[4]0 q[9]2', 'q[3]1 q[4]1', 'q[6]1 q[11]3', 'q[6]2 q[7]3', 'q[1]2 q[6]3', 'q[8]2 q[13]2', 'q[13]3 q[14]1', 'q[3]2 q[8]3', 'q[14]2 q[19]2'], ['q[10]0 q[11]0', 'q[5]0 q[10]1', 'q[11]1 q[12]1', 'q[17]0 q[18]0', 'q[20]0 q[21]0', 'q[16]0 q[21]1', 'q[11]2 q[16]1', 'q[15]0 q[16]2', 'q[15]1 q[20]1', 'q[10]2 q[15]2', 'q[21]2 q[22]0', 'q[18]1 q[23]0', 'q[18]2 q[19]0', 'q[13]0 q[18]3', 'q[12]2 q[13]1', 'q[22]1 q[23]1', 'q[17]1 q[22]2', 'q[16]3 q[17]2', 'q[12]3 q[17]3', 'q[19]1 q[24]0', 'q[23]2 q[24]1']]
cuts = [('q[5]0 q[10]1', 'q[0]1 q[5]1'), ('q[7]1 q[12]0', 'q[11]1 q[12]1'), ('q[11]2 q[16]1', 'q[6]1 q[11]3'), ('q[12]2 q[13]1', 'q[8]2 q[13]2'), ('q[19]1 q[24]0', 'q[14]2 q[19]2')]
num_cuts = len(cuts)
num_subcircuits = len(subcircuits)
cut_subcircuits = []
for cut in cuts:
    for idx, subcircuit in enumerate(subcircuits):
        if cut[0] in subcircuit:
            left = idx
        if cut[1] in subcircuit:
            right = idx
    cut_subcircuits.append((left, right))
print(cut_subcircuits)