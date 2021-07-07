#%%
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

num_qubits = 3
uptowhatK = 100
sdp_tolerance_bound = 0

#Generate initial state
random_generator = np.random.default_rng(123)
initial_state = acp.Initialstate(num_qubits, "efficient_SU2", random_generator, 1)

random_selection_new = True
if random_selection_new == True:
    numberofnewstatestoadd = 3 #Only will be used if 'random_selection_new' is selected

#%% IBMQ STUFF
if use_qiskit:
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

#Here, we use fujiboy's hamiltonian, H = epsilon sum_i Z_i Z_{i+1} + g sum_i X_i
#Here, fujiboy uses epsilon = 1/2
def generate_fuji_boy_hamiltonian(num_qubits, g):
    epsilon = 0.5
    if num_qubits == 1:
        hamiltonian = hcp.generate_arbitary_hamiltonian(num_qubits,[epsilon,g],['3','1'])
    else:
        hamiltonian = hcp.transverse_ising_model_1d(num_qubits, -0.5, g)
    return hamiltonian

def generate_fuji_boy_gamma_and_Lterms(num_qubits):
    gammas_to_append = 0.1
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

def plot_theoretical_expectation_curves(g_min,g_max, observable_obj_list):
    g_vals = np.linspace(g_min, g_max, 50)
    results = dict()
    for g in g_vals:
        hamiltonian = generate_fuji_boy_hamiltonian(num_qubits, g)
        gammas, L_terms = generate_fuji_boy_gamma_and_Lterms(num_qubits)
        qtp_hamiltonian = qutip.Qobj(hamiltonian.to_matrixform())
        qtp_Lterms = [qutip.Qobj(i.to_matrixform()) for i in L_terms]
        qtp_C_ops = [np.sqrt(gammas[i]) * qtp_Lterms[i] for i in range(len(qtp_Lterms))]
        qtp_rho_ss = qutip.steadystate(qtp_hamiltonian, qtp_C_ops)
        #compute the theoretical observable expectation values
        observable_matrixforms = [observable.to_matrixform() for observable in observable_obj_list]
        theoretical_expectation_values = [np.trace(qtp_rho_ss @ observable_matform) for observable_matform in observable_matrixforms]
        results[g] = theoretical_expectation_values
    return results

g = 1
def big_ass_loop(g, observable_obj_list):
    """
    Here, observable_obj_list refers to a list of observable objects
    """
    hamiltonian = generate_fuji_boy_hamiltonian(num_qubits, g)
    gammas, L_terms = generate_fuji_boy_gamma_and_Lterms(num_qubits)
    ansatz = acp.initial_ansatz(num_qubits)

    #function to evaluate the rho_dot in the linblad master eqn
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

    #%%
    #get the steady state using qutip(lol)
    qtp_hamiltonian = qutip.Qobj(hamiltonian.to_matrixform())
    qtp_Lterms = [qutip.Qobj(i.to_matrixform()) for i in L_terms]
    qtp_C_ops = [np.sqrt(gammas[i]) * qtp_Lterms[i] for i in range(len(qtp_Lterms))]
    qtp_rho_ss = qutip.steadystate(qtp_hamiltonian, qtp_C_ops)

    #compute the theoretical observable expectation values
    observable_matrixforms = [observable.to_matrixform() for observable in observable_obj_list]
    theoretical_expectation_values = [np.trace(qtp_rho_ss @ observable_matform) for observable_matform in observable_matrixforms]


    #%%
    #compute GQAS matrices
    fidelity_results = dict()
    observable_expectation_results = dict()
    for k in range(1, uptowhatK + 1):
        print('##########################################')
        print('K = ' +str(k))
        #Generate Ansatz for this round
        if random_selection_new:
            ansatz = acp.gen_next_ansatz(ansatz, hamiltonian, num_qubits,method='random_selection_new',num_new_to_add=numberofnewstatestoadd)
        else:
            ansatz = acp.gen_next_ansatz(ansatz, hamiltonian, num_qubits)
        O_matrices_uneval = []
        for observable in observable_obj_list:
            O_matrix_uneval = mcp.unevaluatedmatrix(num_qubits, ansatz, observable, "O")
            O_matrices_uneval.append(O_matrix_uneval)
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
            O_matrices_evaluated = [i.evaluate_matrix_with_qiskit_circuits(expectation_calculator) for i in O_matrices_uneval]
        else:
            E_mat_evaluated = E_mat_uneval.evaluate_matrix_by_matrix_multiplicaton(initial_state)
            D_mat_evaluated = D_mat_uneval.evaluate_matrix_by_matrix_multiplicaton(initial_state)
            O_matrices_evaluated = [i.evaluate_matrix_by_matrix_multiplicaton(initial_state) for i in O_matrices_uneval]
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

        ##########################################
        #Start of the classical post-processing. #
        ##########################################
        if optimizer == 'feasibility_sdp':
            IQAE_instance = pp.IQAE_Lindblad(num_qubits, D_mat_evaluated, E_mat_evaluated,R_matrices = R_mats_evaluated,F_matrices = F_mats_evaluated,gammas = gammas)
        else:
            IQAE_instance = pp.IQAE_Lindblad(num_qubits, D_mat_evaluated, E_mat_evaluated)


        IQAE_instance.define_optimizer(optimizer, eigh_invcond=eigh_inv_cond,eig_invcond=eig_inv_cond,degeneracy_tol=degeneracy_tol,sdp_tolerance_bound=sdp_tolerance_bound)

        IQAE_instance.evaluate()
        # IQAE_instance.evaluate(kh_test=False)
        #all_energies,all_states = IQAE_instance.get_results_all()
        density_mat,groundstateenergy = IQAE_instance.get_density_matrix_results()
        if type(density_mat) == type(None):
            print('SDP failed for this run, probably due to not high enough K')
        else:
            IQAE_instance.check_if_valid_density_matrix()
            #print(all_energies)
            #print(all_states)
            print("the trace of the beta matrix is", np.trace(density_mat @ E_mat_evaluated))
            print('The ground state energy is\n',groundstateenergy)
            #print('The density matrix is\n',density_mat)
            if IQAE_instance.check_if_hermitian() == True:
                denmat_values,denmat_vects = scp.linalg.eigh(density_mat)
            else:
                denmat_values,denmat_vects = scp.linalg.eig(density_mat)
            denmat_values = np.real(np.round(denmat_values,6))
            #print(np.imag(denmat_values))
            print("the sorted density matrix (beta matrix) eigenvalues are\n",np.sort(denmat_values))

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
            print("trace rho is", np.trace(rho))
            #now, we check if rho (the actual denmat) gives 0 for the linblad master equation

            rho_dot = evaluate_rho_dot(rho, hamiltonian, gammas, L_terms) #should be 0
            # print('rho_dot is: ' + str(rho_dot))
            print('Max value rho_dot is: ' + str(np.max(np.max(rho_dot))))
            qtp_rho = qutip.Qobj(rho)
            fidelity = qutip.metrics.fidelity(qtp_rho, qtp_rho_ss)
            print("The fidelity is", fidelity)
            observable_expectation_results[k] = [np.trace(density_mat @ O_mat_eval) for O_mat_eval in O_matrices_evaluated]
            fidelity_results[k] = fidelity
            if round(fidelity, 6) == 1:
                print("breaking loop as fidelity = 1 already")
                break
    return (observable_expectation_results, theoretical_expectation_values, fidelity_results)

#%% the main chunk
import pickle
import os
if not os.path.exists('pickled_objs'):
    os.mkdir('pickled_objs')

def save_obj(obj, name):
    #name is a string
    with open('pickled_objs/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    #name is a string
    with open('pickled_objs/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

load_prev = False

if load_prev == True:
    result_fname = None #fill this up
    theoretical_curves_fname = None # fill this up
    observable_expectation_results, theoretical_expectation_values, fidelity_results = load_obj(result_fname) 
    theoretical_curves = load_obj(theoretical_curves_fname)
else:
    observable_one = hcp.generate_arbitary_observable(num_qubits, [1], ["1" + "0"*(num_qubits-1)])
    observable_two = hcp.generate_arbitary_observable(num_qubits, [1], ["2" + "0"*(num_qubits-1)])
    observable_three = hcp.generate_arbitary_observable(num_qubits, [1], ["3" + "0"*(num_qubits-1)])
    observables_list = [observable_one, observable_two, observable_three]

    observable_expectation_results, theoretical_expectation_values, fidelity_results = big_ass_loop(g, observables_list)

    # print(observable_expectation_results)
    # print(theoretical_expectation_values)
    # print(fidelity_results)
    g_vals = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    results = [big_ass_loop(g, observables_list) for g in g_vals]
    theoretical_curves = plot_theoretical_expectation_curves(min(g_vals), max(g_vals), observables_list)

    save_obj(results, str(num_qubits) + " results")
    save_obj(theoretical_curves, str(num_qubits) + " theoretical_curves")