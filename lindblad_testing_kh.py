#%%
import numpy as np
from numpy.core.numeric import tensordot 
import ansatz_class_package as acp 
import pauli_class_package as pcp 
import hamiltonian_class_package as hcp 
import matrix_class_package as mcp 
import post_processing as pp
import scipy as scp
import scipy.io

optimizer = 'feasibility_sdp'#'eigh' , 'eig', 'sdp','feasibility_sdp'
eigh_inv_cond = 10**(-6)
eig_inv_cond = 10**(-6)
degeneracy_tol = 5
use_qiskit = False
loadmatlabmatrix = False
runSDPonpython = True

if optimizer == 'feasibility_sdp':
    num_qubits = 3
    uptowhatK = 2
    sdp_tolerance_bound = 0
else:
    num_qubits = 2
    uptowhatK = 2

#Generate initial state
initial_state = acp.Initialstate(num_qubits, "efficient_SU2", 267, 1)

#%% IBMQ STUFF
import Qiskit_helperfunctions as qhf #IBMQ account is loaded here in this import
hub, group, project = "ibm-q-nus", "default", "reservations"

#IMBQ account is loaded in the importing of Qiskit_helperfunctions for now, so this part is commented out (KH, 5 may 2021)
#load IBMQ account. This step is needed if you want to run on the actual quantum computer

#Other parameters for running on the quantum computer. Choose 1 to uncomment.
# sim = "noiseless_qasm"
# num_shots = 30000 #max is 1000000

sim = "noisy_qasm"
quantum_com = "ibmq_rome" #which quantum computer to take the noise profile from
num_shots = 8192 #max is 8192

# sim = "real"
# quantum_com = "ibmq_rome" #which quantum computer to actually run on
# num_shots = 8192 #max is 8192

quantum_computer_choice_results = qhf.choose_quantum_computer(hub, group, project, quantum_com)

#Example on how to create artificial noise model
#couplingmap = [[0,1],[1,2],[2,3],[3,4]]
#quantum_computer_choice_results = qhf.create_quantum_computer_simulation(couplingmap,depolarizingnoise=True,depolarizingnoiseparameter=0.03,bitfliperror=True,bitfliperrorparameter=0.03,measerror=True,measerrorparameter=0.03)

mitigate_meas_error = True 
meas_filter = qhf.measurement_error_mitigator(num_qubits, sim, quantum_computer_choice_results, shots = num_shots)

# mitigate_meas_error = False
# meas_filter = None

#expectation calculator here is an object that has a method that takes in a paulistring object P, and returns a <psi|P|psi>.
#This expectation calculator also stores previously calculated expectation values, so one doesn't need to compute the same expectation value twice.
expectation_calculator = qhf.expectation_calculator(initial_state, sim, quantum_computer_choice_results, meas_error_mitigate = mitigate_meas_error, meas_filter = meas_filter)

if optimizer == 'feasibility_sdp':
    delta = 0.1
    #gammas = [0.1]
    gammas = []
    epsilon = 0.5
    #hamiltonian = hcp.generate_arbitary_hamiltonian(num_qubits,[delta,epsilon],['3','1'])
    #L_terms = [hcp.generate_arbitary_hamiltonian(num_qubits,[1,-1j],['1','2'])]
    hcoeffs = []
    hstrings = []
    if num_qubits == 1:
        hcoeffs.append(0.5)
        hstrings.append("3")
        hstrings.append("1")
        gammas.append(0.1)
    for i in range(num_qubits-1):
        hcoeffs.append(0.5)
        hstrings.append('0'*i+'33'+'0'*(num_qubits-2-i))
    for i in range(num_qubits):
        hcoeffs.append(0.5)
        hstrings.append('0'*i+'1'+'0'*(num_qubits-1-i))
    hamiltonian = hcp.generate_arbitary_hamiltonian(num_qubits,hcoeffs,hstrings)

    L_terms = []
    for i in range(num_qubits):
        gammas.append(0.1)
        L_terms.append(hcp.generate_arbitary_hamiltonian(num_qubits,[1],['0'*i+'3'+'0'*(num_qubits-1-i)]))
        gammas.append(0.1)
        L_terms.append(hcp.generate_arbitary_hamiltonian(num_qubits,[0.5,-0.5j],['0'*i+'1'+'0'*(num_qubits-1-i),'0'*i+'2'+'0'*(num_qubits-1-i)]))
    if num_qubits == 1:
        gammas.append(0.1)
        L_terms.append(hcp.generate_arbitary_hamiltonian(1, [0.5,0.5j],["1","2"]))
else:    
    # Going into the Fock-Liouville space, old code
    L = np.array([[-0.1,-0.25j,0.25j,0],[-0.25j,-0.05-0.1j,0,0.25j],[0.25j,0,-0.05+0.1j,-0.25j],[0.1,0.25j,-0.25j,0]])
    #Converting L^dag L into Hamiltonian
    LdagL = np.array([[0.145,0.025+0.0375j,0.025-0.0375j,-0.125],[0.025-0.0375j,0.1375,-0.125,-0.025-0.0125j],[0.025+0.0375j,-0.125,0.1375,-0.025+0.0125j],[-0.125,-0.025+0.0125j,-0.025-0.0125j,0.125]])
    pauli_decomp = pcp.paulinomial_decomposition(LdagL) 
    # print(list(pauli_decomp.values()))
    hamiltonian = hcp.generate_arbitary_hamiltonian(num_qubits, list(pauli_decomp.values()), list(pauli_decomp.keys()))

ansatz = acp.initial_ansatz(num_qubits)

#compute GQAS matrices
for k in range(1, uptowhatK + 1):
    print('##########################################')
    print('K = ' +str(k))
    #Generate Ansatz for this round
    ansatz = acp.gen_next_ansatz(ansatz, hamiltonian, num_qubits)

    E_mat_uneval = mcp.unevaluatedmatrix(num_qubits, ansatz, hamiltonian, "E")
    D_mat_uneval = mcp.unevaluatedmatrix(num_qubits, ansatz, hamiltonian, "D")

    if optimizer == 'feasibility_sdp':
        R_mats_uneval = []
        F_mats_uneval = []
        for thisL in L_terms:
            R_mats_uneval.append(mcp.unevaluatedmatrix(num_qubits,ansatz,thisL,"D"))
            thisLdagL = hcp.multiply_hamiltonians(hcp.dagger_hamiltonian(thisL),thisL)
            F_mats_uneval.append(mcp.unevaluatedmatrix(num_qubits,ansatz,thisLdagL,"D"))
    
            

    #Here is where we should be able to specify how to evaluate the matrices.
    #However only the exact method (classical matrix multiplication) has been
    #implemented so far
    if use_qiskit:
        E_mat_evaluated = E_mat_uneval.evaluate_matrix_with_qiskit_circuits(expectation_calculator)
        D_mat_evaluated = D_mat_uneval.evaluate_matrix_with_qiskit_circuits(expectation_calculator)
    else:
        E_mat_evaluated = E_mat_uneval.evaluate_matrix_by_matrix_multiplicaton(initial_state)
        D_mat_evaluated = D_mat_uneval.evaluate_matrix_by_matrix_multiplicaton(initial_state)
    if optimizer == 'feasibility_sdp':
        R_mats_evaluated = []
        for r in R_mats_uneval:
            if use_qiskit:
                R_mats_evaluated.append(r.evaluate_matrix_with_qiskit_circuits(expectation_calculator))
            else:
                R_mats_evaluated.append(r.evaluate_matrix_by_matrix_multiplicaton(initial_state))
        F_mats_evaluated = []
        for f in F_mats_uneval:
            if use_qiskit:
                F_mats_evaluated.append(f.evaluate_matrix_with_qiskit_circuits(expectation_calculator))
            else:
                F_mats_evaluated.append(f.evaluate_matrix_by_matrix_multiplicaton(initial_state))


    #Save matrices for testing with matlab
    if optimizer == 'feasibility_sdp':
        scipy.io.savemat("Jonstufftesting/Emat" +str(k) +".mat",{"E": E_mat_evaluated,"D":D_mat_evaluated,"R":R_mats_evaluated,"F":F_mats_evaluated})
        print('Matrices have been generated, saved in Jonstufftestingfolder.')
    #print(D_mat_evaluated)
    ##########################################
    #Start of the classical post-processing. #
    ##########################################
    if optimizer == 'feasibility_sdp':
        IQAE_instance = pp.IQAE_Lindblad(num_qubits, D_mat_evaluated, E_mat_evaluated,R_matrices = R_mats_evaluated,F_matrices = F_mats_evaluated,gammas = gammas)
    else:
        IQAE_instance = pp.IQAE_Lindblad(num_qubits, D_mat_evaluated, E_mat_evaluated)


    IQAE_instance.define_optimizer(optimizer, eigh_invcond=eigh_inv_cond,eig_invcond=eig_inv_cond,degeneracy_tol=degeneracy_tol,sdp_tolerance_bound=sdp_tolerance_bound)

    if optimizer == 'feasibility_sdp' and runSDPonpython == False:
        print('NOT RUNNING SDP ON PYTHON. JUST USING FIRST PART OF CODE TO GENERATE ANSATZ FOR 2ND PART')
    else:
        IQAE_instance.evaluate()
        #all_energies,all_states = IQAE_instance.get_results_all()
        density_mat,groundstateenergy = IQAE_instance.get_density_matrix_results()
        if type(density_mat) == type(None):
            print('SDP failed for this run, probably due to not high enough K')
        else:
            IQAE_instance.check_if_valid_density_matrix()
            #print(all_energies)
            #print(all_states)
            print('The ground state energy is\n',groundstateenergy)
            #print('The density matrix is\n',density_mat)
            if IQAE_instance.check_if_hermitian() == True:
                denmat_values,denmat_vects = scp.linalg.eigh(density_mat)
            else:
                denmat_values,denmat_vects = scp.linalg.eig(density_mat)
            denmat_values = np.real(np.round(denmat_values,6))
            #print(np.imag(denmat_values))
            print("the sorted density matrix (beta matrix) eigenvalues are\n",np.sort(denmat_values))
            #print("the density matrix eigenvectors are\n",denmat_vects)
#%%
#testing for the feasibility routine

'''NOTES FOR SELF: Right now matlab functionality is not built into this. So, first time you run, this python file will generate the D, E, R, and F matrices. Ignore everything else. Then, go to matlab and run sdp.m . 
That will generate the beta matrix. Then, run THIS same file again with loadmatlabmatrix = True. This file will still do the generating of matrices ect, but now for the 2nd half (checking) it will use the saved matlab matrix.'''

if loadmatlabmatrix == True:
    print('LOADING MATRICES THAT SHOULD HAVE BEEN GENERATED FROM MATLAB. ENSURE THIS IS DONE.')
    density_mat = scipy.io.loadmat('Jonstufftesting/'+'savedmatrixfrommatlab.mat')['betarho']

p_string_matrices = [i.get_paulistring().get_matrixform() for i in ansatz.get_moments()]
ini_statevec_vecform = initial_state.get_statevector()
csk_states = [i@ini_statevec_vecform for i in p_string_matrices]
rho = np.zeros(shape=(2**num_qubits,2**num_qubits), dtype = np.complex128)
trace = 0
for i in range(len(density_mat)):
    for j in range(len(density_mat)):
        i_j_entry = density_mat[(i,j)]
        i_j_ketbra = np.outer(csk_states[i], csk_states[j].conj().T)
        rho += i_j_entry * i_j_ketbra
        trace += i_j_entry * csk_states[j].conj().T @ csk_states[i]

rho_eigvals,rho_eigvecs = scipy.linalg.eigh(rho)        
print('rho_eigvals is: ' + str(rho_eigvals))
#now, we check if rho (the actual denmat) gives 0 for the linblad master equation
def evaluate_rho_dot(rho, hamiltonian_class_object, gammas, L_terms):
    hamiltonian_mat = hamiltonian_class_object.to_matrixform()
    coherent_evo = -1j * (hamiltonian_mat @ rho - rho @ hamiltonian_mat)
    quantum_jumps_total = 0 + 0*1j
    for i in range(len(gammas)):
        gamma_i = gammas[i]
        L_i_mat = L_terms[i].to_matrixform()
        L_i_dag_L_i = L_i_mat.conj().T @ L_i_mat
        anti_commutator = L_i_dag_L_i @ rho + rho @ L_i_dag_L_i
        jump_term = L_i_mat @ rho @ L_i_mat.conj().T
        quantum_jumps_total += gamma_i * (jump_term - 0.5*anti_commutator)
    return coherent_evo + quantum_jumps_total

rho_dot = evaluate_rho_dot(rho, hamiltonian, gammas, L_terms) #should be 0
# print('rho_dot is: ' + str(rho_dot))
print('Max value rho_dot is: ' + str(np.max(np.max(rho_dot))))
#%%
#Testing for the mapping routine. 
# p_string_matrices = [i.get_paulistring().get_matrixform() for i in ansatz.get_moments()]
# ini_statevec_vecform = initial_state.get_statevector()
# csk_states = [i@ini_statevec_vecform for i in p_string_matrices]
# rho_prime = np.zeros(shape=(4,4), dtype = np.complex128)
# trace = 0
# for i in range(len(density_mat)):
#     for j in range(len(density_mat)):
#         i_j_entry = density_mat[(i,j)]
#         #in the bottom line, its j,i instead of i,j because of some bug somewhere. Something accidentally got reversed somewhere, I can't find...
#         i_j_ketbra = np.outer(csk_states[i], csk_states[j].conj().T)
#         rho_prime += i_j_entry * i_j_ketbra
#         trace += i_j_entry * csk_states[j].conj().T @ csk_states[i]

# rho_prime_eigvals,rho_prime_eigvecs = scipy.linalg.eigh(rho_prime)        
# #here, we take the eigvec that corresponds to the non-zero eigval
# rho = np.zeros(shape=(2,2), dtype = np.complex128)
# rho[(0,0)] = rho_prime_eigvecs[:,3][0]
# rho[(0,1)] = rho_prime_eigvecs[:,3][1]
# rho[(1,0)] = rho_prime_eigvecs[:,3][2]
# rho[(1,1)] = rho_prime_eigvecs[:,3][3]
# print("The eigenvalues of rho are", scipy.linalg.eigvalsh(rho))

#%% Old IQAE code that might be useful
#Testing if the IQAE result is a valid density matrix
# ground_state = all_states[0]

# #the ansatz states
# p_matrices = [i.get_paulistring().get_string_for_hash() for i in ansatz.get_moments()]
# # print(p_matrices)
# # p_matrices = ["00", "02", "03", "10", "11", "13", "20", "23", "30", "31", "32", "33"]
# p_matrices_matform = [pcp.get_pauli_string_from_index_string(i) for i in p_matrices]
# ini_statevec = initial_state.get_statevector()
# csk_states = [i @ ini_statevec for i in p_matrices_matform]

# final_state = np.zeros(4) * 1j*np.zeros(4)
# for i in range(len(csk_states)):
#     final_state += ground_state[i]*csk_states[i]

# density_mat = np.empty(shape=(2,2), dtype=np.complex128)   
# density_mat[(0,0)] = final_state[0]
# density_mat[(1,0)] = final_state[1]
# density_mat[(0,1)] = final_state[2]
# density_mat[(1,1)] = final_state[3]
# print("the density matrix is\n", density_mat)
# denmat_values,denmat_vects = scp.linalg.eig(density_mat)
# print("the density matrix eigenvalues are\n",denmat_values)
# print("the density matrix eigenvectors are\n",denmat_vects)
# %%
