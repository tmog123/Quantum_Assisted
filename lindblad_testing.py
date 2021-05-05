import numpy as np 
import ansatz_class_package as acp 
import pauli_class_package as pcp 
import hamiltonian_class_package as hcp 
import matrix_class_package as mcp 
import post_processing as pp

uptowhatK = 2
num_qubits = 2
optimizer = 'eig'
eig_inv_cond = 10**(-6)
degeneracy_tol = 5

#Generate initial state
initial_state = acp.Initialstate(num_qubits, "efficient_SU2", 189, 2)

#define Hamiltonian
gamma = 0.1
delta = 0.1
epsilon = 0.5
hamiltonian = hcp.generate_arbitary_hamiltonian(2,[-1j*delta,1j*delta,-1j*epsilon,1j*epsilon,gamma,-1j*gamma,-1j*gamma,-gamma,-2*gamma],["03","30","01","10","11","21","12","22","00"])
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
    IQAE_instance.define_optimizer(optimizer, eig_invcond=eig_inv_cond,degeneracy_tol=degeneracy_tol)

    IQAE_instance.evaluate()
    all_energies,all_states = IQAE_instance.get_results_all()
    print(all_energies)