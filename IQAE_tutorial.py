import numpy as np 
import ansatz_class_package as acp 
import pauli_class_package as pcp 
import hamiltonian_class_package as hcp 
import matrix_class_package as mcp 
import post_processing as pp
import time

#starting time
starttime = time.time()
#The IQAE algorithm is found in https://arxiv.org/abs/2010.05638

#Parameters
uptowhatK = 3
num_qubits = 4
optimizer = 'eigh' #'eigh' or 'qcqp'
eigh_inv_cond = 10**(-12)

#create initial state
initial_state = acp.Initialstate(num_qubits, "efficient_SU2", 1, 5)

#define Hamiltonian
hamiltonian = hcp.heisenberg_xyz_model(num_qubits)

#create Initial Ansatz for K = 0
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
    IQAE_instance = pp.IQAE(num_qubits, D_mat_evaluated, E_mat_evaluated)
    IQAE_instance.define_optimizer(optimizer, eigh_invcond=eigh_inv_cond)

    IQAE_instance.evaluate()
    ground_state_energy,ground_state_alphavec = IQAE_instance.get_results()
    print("The ground state energy is", ground_state_energy)

#Compare with just diagonalizing matrix
cS_instance = pp.classicalSimulator(num_qubits,initial_state,hamiltonian)
ground_energy,ground_state = cS_instance.find_ground_state()
print('The real ground state energy is '+str(ground_energy))

#end time
endtime = time.time()

#total time taken
print(f"Runtime of the program is {endtime - starttime}")

