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
    #print(all_energies)
    #print(all_states)