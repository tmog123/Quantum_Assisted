import numpy as np
import ansatz_class_package as acp 
import pauli_class_package as pcp 
import hamiltonian_class_package as hcp 
import matrix_class_package as mcp 
import post_processing as pp
import scipy as scp
import scipy.io
import qutip 
optimizer = 'feasibility_sdp'#'eigh' , 'eig', 'sdp','feasibility_sdp'
eigh_inv_cond = 10**(-6)
eig_inv_cond = 10**(-6)
degeneracy_tol = 5
use_qiskit = False
loadmatlabmatrix = False
runSDPonpython = True
num_qubits = 5
uptowhatK = 3
sdp_tolerance_bound = 0

howmanyrandomhamtogenerate = 10
eachhamuptohowmanyterms = 5
seedforhamgeneration = 238
maximumbeta = 1

classicalmethodstotest = ['direct','eigen','power','iterative-gmres','iterative-lgmres','iterative-bicgstab','svd']

listofrandomhamiltonians = hcp.generate_package_of_random_hamiltonians(num_qubits,howmanyrandomhamtogenerate,eachhamuptohowmanyterms,seedforhamgeneration,maximumbeta)

#Generate initial state
random_generator = np.random.default_rng(719)
initial_state = acp.Initialstate(num_qubits, "efficient_SU2", random_generator, 1)

random_selection_new = False
if random_selection_new == True:
    numberofnewstatestoadd = 10 #Only will be used if 'random_selection_new' is selected


#Generate local noise terms
def generate_local_Lterms(num_qubits):
    gammas_to_append = 0.5
    gammas = []
    L_terms = []
    if num_qubits == 1:
        gammas.append(gammas_to_append)
        L_terms.append(hcp.generate_arbitary_hamiltonian(1, [0.5,0.5j],["1","2"]))
    else:
        for i in range(num_qubits):
            gammas.append(gammas_to_append)
            L_terms.append(hcp.generate_arbitary_hamiltonian(num_qubits,[1],['0'*i+'3'+'0'*(num_qubits-1-i)]))
            gammas.append(gammas_to_append)
            L_terms.append(hcp.generate_arbitary_hamiltonian(num_qubits,[0.5,-0.5j],['0'*i+'1'+'0'*(num_qubits-1-i),'0'*i+'2'+'0'*(num_qubits-1-i)]))
    return (gammas, L_terms)

def generate_classical_results(num_qubits,listofrandomhamiltonians,classicalmethods,noise_term_function):
    finalresults = {}
    for met in classicalmethods:
        finalresults[met] = []
    for i in range(len(listofrandomhamiltonians)):
        ham = listofrandomhamiltonians[i]
        gammas, L_terms = noise_term_function(num_qubits)
        qtp_hamiltonian = qutip.Qobj(ham.to_matrixform())
        qtp_Lterms = [qutip.Qobj(i.to_matrixform()) for i in L_terms]
        qtp_C_ops = [np.sqrt(gammas[i]) * qtp_Lterms[i] for i in range(len(qtp_Lterms))]
        for met in classicalmethods:
            qtp_rho_ss = qutip.steadystate(qtp_hamiltonian, qtp_C_ops,method=met)
            finalresults[met].append(qtp_rho_ss)
    return finalresults

if use_qiskit:
    import Qiskit_helperfunctions as qhf #IBMQ account is loaded here in this import
    hub, group, project = "ibm-q-nus", "default", "reservations"

    #IMBQ account is loaded in the importing of Qiskit_helperfunctions for now, so this part is commented out (KH, 5 may 2021)
    #load IBMQ account. This step is needed if you want to run on the actual quantum computer

    #Other parameters for running on the quantum computer. Choose 1 to uncomment.
    # sim = "noiseless_qasm"
    # num_shots = 30000 #max is 1000000

    sim = "noisy_qasm"
    quantum_com = "ibmq_bogota" #which quantum computer to take the noise profile from
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

def generate_IQAE_results(num_qubits,listofrandomhamiltonians,noise_term_function):
    finalresults = []
    for hamiltonian in listofrandomhamiltonians:
        gammas, L_terms = noise_term_function(num_qubits)
        for k in range(1, uptowhatK + 1):
            print('##########################################')
            print('K = ' +str(k))
            # max_k_val = k
            #Generate Ansatz for this round
            if random_selection_new:
                ansatz = acp.gen_next_ansatz(ansatz, hamiltonian, num_qubits,method='random_selection_new',num_new_to_add=numberofnewstatestoadd)
            else:
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
            if optimizer == 'feasibility_sdp':
                IQAE_instance = pp.IQAE_Lindblad(num_qubits, D_mat_evaluated, E_mat_evaluated,R_matrices = R_mats_evaluated,F_matrices = F_mats_evaluated,gammas = gammas)
            else:
                IQAE_instance = pp.IQAE_Lindblad(num_qubits, D_mat_evaluated, E_mat_evaluated)            
            IQAE_instance.define_optimizer(optimizer, eigh_invcond=eigh_inv_cond,eig_invcond=eig_inv_cond,degeneracy_tol=degeneracy_tol,sdp_tolerance_bound=sdp_tolerance_bound)

            IQAE_instance.evaluate()
            density_mat,groundstateenergy = IQAE_instance.get_density_matrix_results()
        #We ONLY append the highest K den mat (should be most accurate result)
        finalresults.append(density_mat)




'''###################################################################'''
'''START HERE'''
#Think we should compare density matrices, so I made functions to get out the qtp.steadystate results (which we can presumably get the density matrices from) and the density matrices from IQAE

#The function below returns a dictionary, with keys being the classical methods, values being the qtp.steadystate results
classicalresults = generate_classical_results(num_qubits,listofrandomhamiltonians,classicalmethodstotest,generate_local_Lterms)
#The function below returns a list of density matrices found using IQAE
IQAEresults = generate_IQAE_results(num_qubits,listofrandomhamiltonians,generate_local_Lterms)

























