import numpy as np 
import ansatz_class_package as acp 
import pauli_class_package as pcp 
import hamiltonian_class_package as hcp 
import matrix_class_package as mcp 
import post_processing as pp

uptowhatK = 1
num_qubits = 2
optimizer = 'eigh'
eigh_inv_cond = 10**(-6)
degeneracy_tol = 5

#Generate initial state
initial_state = acp.Initialstate(num_qubits, "efficient_SU2", 267, 2)

#Converting L^dag L into Hamiltonian
LdagL = np.array([[0.145,0.025+0.0375j,0.025-0.0375j,-0.125],[0.025-0.0375j,0.1375,-0.125,-0.025-0.0125j],[0.025+0.0275j,-0.125,0.1375,-0.025+0.0125j],[-0.125,-0.025+0.0125j,-0.025-0.0125j,0.125]])
pauli_decomp = pcp.paulinomial_decomposition(LdagL) 
hamiltonian = hcp.generate_arbitary_hamiltonian(num_qubits, list(pauli_decomp.values()), list(pauli_decomp.keys()))

print('Beta values are ' + str(hamiltonian.return_betas()))

ansatz = acp.initial_ansatz(num_qubits)

#Run IQAE
for k in range(1, uptowhatK + 1):
    print(k)
    #Generate Ansatz for this round
    ansatz = acp.gen_next_ansatz(ansatz, hamiltonian, num_qubits)

    E_mat_uneval = mcp.unevaluatedmatrix(num_qubits, ansatz, hamiltonian, "E")
    D_mat_uneval = mcp.unevaluatedmatrix(num_qubits, ansatz, hamiltonian, "D")

    #Here is where we should be able to specify how to evaluate the matrices. However only the exact method (classical matrix multiplication) has been implemented so far
    E_mat_evaluated = E_mat_uneval.evaluate_matrix_by_matrix_multiplicaton(initial_state)
    D_mat_evaluated = D_mat_uneval.evaluate_matrix_by_matrix_multiplicaton(initial_state)

    ##########################################
    #Start of the classical post-processing. #
    ##########################################
    IQAE_instance = pp.IQAE_Lindblad(num_qubits, D_mat_evaluated, E_mat_evaluated)
    IQAE_instance.define_optimizer(optimizer, eigh_invcond=eigh_inv_cond,degeneracy_tol=degeneracy_tol)

    IQAE_instance.evaluate()
    all_energies,all_states = IQAE_instance.get_results_all()
    print(all_energies)
    print(all_states)


#Testing if the IQAE result is a valid density matrix
ground_state = all_states[1]

#the ansatz states
p_matrices = [i.get_paulistring().get_string_for_hash() for i in ansatz.get_moments()]
# print(p_matrices)
# p_matrices = ["00", "02", "03", "10", "11", "13", "20", "23", "30", "31", "32", "33"]
p_matrices_matform = [pcp.get_pauli_string_from_index_string(i) for i in p_matrices]
ini_statevec = initial_state.get_statevector()
csk_states = [i @ ini_statevec for i in p_matrices_matform]

final_state = np.zeros(4) * 1j*np.zeros(4)
for i in range(len(csk_states)):
    final_state += ground_state[i]*csk_states[i]

density_mat = np.empty(shape=(2,2), dtype=np.complex128)   
density_mat[(0,0)] = final_state[0]
density_mat[(1,0)] = final_state[1]
density_mat[(0,1)] = final_state[2]
density_mat[(1,1)] = final_state[3]
print("the density matrix is", density_mat)