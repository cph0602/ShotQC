from helper_functions.utils import find_distinct_q_values
subcircuits = [['q[4]0 q[6]0', 'q[4]1 q[6]1', 'q[4]2 q[11]0', 'q[4]3 q[11]1', 'q[4]4 q[2]0', 'q[4]5 q[2]1', 'q[6]2 q[11]2', 'q[6]3 q[11]3', 'q[6]4 q[13]0', 'q[6]5 q[13]1', 'q[12]0 q[13]2', 'q[12]1 q[13]3', 'q[12]2 q[1]0', 'q[12]3 q[1]1', 'q[13]4 q[11]4', 'q[13]5 q[11]5', 'q[2]2 q[14]2', 'q[2]3 q[14]3', 'q[12]4 q[15]0', 'q[12]5 q[15]1', 'q[2]4 q[15]2', 'q[2]5 q[15]3', 'q[14]4 q[15]4', 'q[14]5 q[15]5'], ['q[3]0 q[7]0', 'q[3]1 q[7]1', 'q[7]2 q[5]0', 'q[7]3 q[5]1', 'q[3]2 q[9]0', 'q[3]3 q[9]1', 'q[3]4 q[0]0', 'q[3]5 q[0]1', 'q[7]4 q[10]0', 'q[7]5 q[10]1', 'q[5]2 q[10]2', 'q[5]3 q[10]3', 'q[10]4 q[8]0', 'q[10]5 q[8]1', 'q[8]2 q[9]2', 'q[8]3 q[9]3', 'q[9]4 q[0]2', 'q[9]5 q[0]3', 'q[5]4 q[1]2', 'q[5]5 q[1]3', 'q[1]4 q[0]4', 'q[1]5 q[0]5', 'q[8]4 q[14]0', 'q[8]5 q[14]1']]

cuts = [('q[12]3 q[1]1', 'q[5]4 q[1]2'), ('q[8]5 q[14]1', 'q[2]2 q[14]2')]

num_qubits = 16
subcircuits_output = []
for subcircuit in subcircuits:
    subcircuits_output.append(find_distinct_q_values(subcircuit))
print(subcircuits_output)
num_cuts = len(cuts)
num_subcircuits = len(subcircuits)
cut_subcircuits = []
cut_locations = []
cut_resolve = {}
q_cuts = []
for cut in cuts:
    left_qubits = find_distinct_q_values([cut[0]])
    right_qubits = find_distinct_q_values([cut[1]])
    cut_loc = list(set(left_qubits)&set(right_qubits))[0]
    cut_locations.append(list(set(left_qubits)&set(right_qubits))[0])
    for idx, subcircuit in enumerate(subcircuits):
        if cut[0] in subcircuit:
            left = idx
        if cut[1] in subcircuit:
            right = idx
    for i, q in enumerate(subcircuits_output[left]):
        if q == cut_loc:
            l_loc = i
            break
    for i, q in enumerate(subcircuits_output[right]):
        if q == cut_loc:
            r_loc = i
            break
    cut_subcircuits.append((left, right))
    q_cuts.append(((left, l_loc), (right, r_loc)))
    cut_resolve[cut_loc] = right
print(cut_subcircuits)
print(cut_locations)
output_q = 0
mapping = []
org_to_new = {}
for idx in range(num_subcircuits):
    for q in subcircuits_output[idx]:
        is_cut = False
        for j, cut_q in enumerate(cut_locations):
            if q == cut_q:
                if idx == cut_subcircuits[j][1]:
                    org_to_new[q] = output_q
                    output_q += 1
                is_cut = True
                break
        if not is_cut:
            org_to_new[q] = output_q
            output_q += 1

for i in range(num_qubits):
    mapping.append(org_to_new[i])
print(mapping)
print(q_cuts)