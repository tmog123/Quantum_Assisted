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
import plotting_package as plotp

optimizer = 'feasibility_sdp'#'eigh' , 'eig', 'sdp','feasibility_sdp'
eigh_inv_cond = 10**(-6)
eig_inv_cond = 10**(-6)
degeneracy_tol = 5
use_qiskit = False
loadmatlabmatrix = False
runSDPonpython = True

num_qubits = 3
uptowhatK = 3
sdp_tolerance_bound = 0

Gamma = 0.9
mu = 0.5

#Generate initial state
random_generator = np.random.default_rng(497)
initial_state = acp.Initialstate(num_qubits, "efficient_SU2", random_generator, 1)

random_selection_new = True
if random_selection_new == True:
    numberofnewstatestoadd = 10 #Only will be used if 'random_selection_new' is selected

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

#Here, we use fujiboy's hamiltonian, H = epsilon sum_i Z_i Z_{i+1} + g sum_i X_i
#Here, fujiboy uses epsilon = 1/2
def generate_XXZ_hamiltonian(num_qubits, delta):
    #epsilon = 0.5
    if num_qubits == 1:
        raise(RuntimeError('Cannot generate Hamiltonian with 1 qubit'))
    else:
        hamiltonian = hcp.heisenberg_xyz_model(num_qubits, jx = 1, jy = 1, jz = delta)
    return hamiltonian

def generate_nonlocaljump_gamma_and_Lterms(num_qubits,Gamma,mu):
    #gammas_to_append = 1
    gammas = []
    L_terms = []
    if num_qubits == 1:
        raise(RuntimeError('Cannot generate non-local jump terms with 1 qubit'))
    else:
        gammas.append(1)
        cof = np.sqrt(Gamma*(1-mu))
        L_terms.append(hcp.generate_arbitary_hamiltonian(num_qubits,[0.25*cof,0.25j*cof,-0.25j*cof,0.25*cof],['1'+'0'*(num_qubits-2)+'1','2'+'0'*(num_qubits-2)+'1','1'+'0'*(num_qubits-2)+'2','2'+'0'*(num_qubits-2)+'2']))
        gammas.append(1)
        cof = np.sqrt(Gamma*(1+mu))
        L_terms.append(hcp.generate_arbitary_hamiltonian(num_qubits,[0.25*cof,-0.25j*cof,0.25j*cof,0.25*cof],['1'+'0'*(num_qubits-2)+'1','2'+'0'*(num_qubits-2)+'1','1'+'0'*(num_qubits-2)+'2','2'+'0'*(num_qubits-2)+'2']))
    return (gammas, L_terms)

def plot_theoretical_expectation_curves(delta_min,delta_max,Gamma,mu, observable_obj_list):
    delta_vals = np.linspace(delta_min, delta_max, 50)
    results = dict()
    for delta in delta_vals:
        hamiltonian = generate_XXZ_hamiltonian(num_qubits, delta)
        gammas, L_terms = generate_nonlocaljump_gamma_and_Lterms(num_qubits,Gamma,mu)
        qtp_hamiltonian = qutip.Qobj(hamiltonian.to_matrixform())
        qtp_Lterms = [qutip.Qobj(i.to_matrixform()) for i in L_terms]
        qtp_C_ops = [np.sqrt(gammas[i]) * qtp_Lterms[i] for i in range(len(qtp_Lterms))]
        qtp_rho_ss = qutip.steadystate(qtp_hamiltonian, qtp_C_ops,method="svd")
        #compute the theoretical observable expectation values
        observable_matrixforms = [observable.to_matrixform() for observable in observable_obj_list]
        theoretical_expectation_values = [np.trace(qtp_rho_ss.full() @ observable_matform) for observable_matform in observable_matrixforms]
        results[delta] = theoretical_expectation_values
    keys = list(results.keys())
    values = list(results.values())
    values_transposed = list(zip(*values)) 
    return (keys,values_transposed) #this is in a plottable form
    # return results

def big_ass_loop(delta,Gamma,mu, observable_obj_list):
    """
    Here, observable_obj_list refers to a list of observable objects
    """
    hamiltonian = generate_XXZ_hamiltonian(num_qubits, delta)
    gammas, L_terms = generate_nonlocaljump_gamma_and_Lterms(num_qubits,Gamma,mu)
    ansatz = acp.initial_ansatz(num_qubits)
    #get the steady state using qutip(lol)
    qtp_hamiltonian = qutip.Qobj(hamiltonian.to_matrixform())
    qtp_Lterms = [qutip.Qobj(i.to_matrixform()) for i in L_terms]
    qtp_C_ops = [np.sqrt(gammas[i]) * qtp_Lterms[i] for i in range(len(qtp_Lterms))]
    qtp_rho_ss = qutip.steadystate(qtp_hamiltonian, qtp_C_ops,method='iterative-gmres')

    #compute the theoretical observable expectation values
    observable_matrixforms = [observable.to_matrixform() for observable in observable_obj_list]
    theoretical_expectation_values = [np.trace(qtp_rho_ss.full() @ observable_matform) for observable_matform in observable_matrixforms]

    #%%
    #compute GQAS matrices
    fidelity_results = dict()
    observable_expectation_results = dict()
    for k in range(1, uptowhatK + 1):
        print('##########################################')
        print('K = ' +str(k))
        # max_k_val = k
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
        result_dictionary = pp.analyze_density_matrix(num_qubits,initial_state,IQAE_instance,E_mat_evaluated,ansatz,hamiltonian,gammas,L_terms,qtp_rho_ss,O_matrices_evaluated)
        observable_expectation_results[k] = result_dictionary['observable_expectation']
        fidelity_results[k] = result_dictionary['fidelity']
            #if round(fidelity, 6) == 1:
            #    print("breaking loop as fidelity = 1 already")
            #    #raise(RuntimeError("Fidelity = 1!"))
            #    break
    #print('JON: Got %s results'%len(fidelity_results))
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
    print("loading previously computed results for " + str(num_qubits) + " qubits")
    result_fname = None #fill this up
    theoretical_curves_fname = None # fill this up
    result_fname = str(num_qubits)+"_qubits_results"
    theoretical_curves_fname = str(num_qubits)+"_qubits_theoretical_curves"
    # observable_expectation_results, theoretical_expectation_values, fidelity_results = load_obj(result_fname) 
    results = load_obj(result_fname)
    theoretical_curves = load_obj(theoretical_curves_fname)
else:
    observable_one = hcp.generate_arbitary_observable(num_qubits, [1], ["1" + "0"*(num_qubits-1)])
    observable_two = hcp.generate_arbitary_observable(num_qubits, [1], ["2" + "0"*(num_qubits-1)])
    observable_three = hcp.generate_arbitary_observable(num_qubits, [1], ["3" + "0"*(num_qubits-1)])
    observables_list = [observable_one, observable_two, observable_three]

    # observable_expectation_results, theoretical_expectation_values, fidelity_results = big_ass_loop(g, observables_list)

    delta_vals = [0,0.125,0.25,0.375, 0.5,0.625,0.75,0.875, 1.0, 1.5, 2.0]
    #g_vals = [0.5]
    results = {delta:big_ass_loop(delta,Gamma=Gamma,mu=mu,observable_obj_list= observables_list) for delta in delta_vals}
    theoretical_curves = plot_theoretical_expectation_curves(min(delta_vals), max(delta_vals),Gamma,mu, observables_list)


    if use_qiskit == True:
        fname_append = str(num_qubits) + "_qubits" + "_sim=" + str(sim) + "_"
    else:
        fname_append = str(num_qubits) + "_qubits" + "_"

    save_obj(results, fname_append + "results")
    save_obj(theoretical_curves, fname_append + "theoretical_curves")

#%% plot stuff
#theres' a small bug here lol. hmm
import matplotlib.pyplot as plt 

if random_selection_new:
    num_of_csk_states = lambda k: numberofnewstatestoadd * k + 1

plotp.plot_fidelities(num_qubits,results,random_selection_new,num_of_csk_states)
plotp.print_plot('graphsforpaper/XXZ_%s_qubit_fidelity.png'%num_qubits,bboxtight="tight")
plotp.qutip_comparison_with_k_plot_expectation_values(num_qubits,results, theoretical_curves, [3,4],random_selection_new,num_of_csk_states)
plotp.print_plot('graphsforpaper/XXZ_%s_qubit.png'%num_qubits,bboxtight="tight")

# %%
