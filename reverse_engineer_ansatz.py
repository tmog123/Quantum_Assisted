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
import matplotlib.pyplot as plt
import qiskit.quantum_info as qi

g_vals = [0,0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
supportlevel = 0.01
num_qubits = 5
# optimizer = 'feasibility_sdp'#'eigh' , 'eig', 'sdp','feasibility_sdp'
eigh_inv_cond = 10**(-6)
eig_inv_cond = 10**(-6)
degeneracy_tol = 5
use_qiskit = False
loadmatlabmatrix = False
# runSDPonpython = True
num_qubits = 5
uptowhatK = 5
sdp_tolerance_bound = 0
what_starting_state = 'Random'# 'Random', 'Ground_state', 'Random_statevector'
if what_starting_state == 'Random':
    random_generator = np.random.default_rng(123)
    initial_state = acp.Initialstate(num_qubits, "efficient_SU2", random_generator, 1)
random_selection_new = True
if random_selection_new == True:
    numberofnewstatestoadd = 10 #Only will be used if 'random_selection_new' is selected

def generate_fuji_boy_hamiltonian(num_qubits, g):
    epsilon = 0.5
    if num_qubits == 1:
        hamiltonian = hcp.generate_arbitary_hamiltonian(num_qubits,[epsilon,g],['3','1'])
    else:
        hamiltonian = hcp.transverse_ising_model_1d(num_qubits, -0.5, g)
    return hamiltonian

def generate_fuji_boy_gamma_and_Lterms(num_qubits):
    gammas_to_append = 1
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
        theoretical_expectation_values = [np.trace(qtp_rho_ss.full() @ observable_matform) for observable_matform in observable_matrixforms]
        results[g] = theoretical_expectation_values
    keys = list(results.keys())
    values = list(results.values())
    values_transposed = list(zip(*values)) 
    return (keys,values_transposed) #this is in a plottable form

observable_one = hcp.generate_arbitary_observable(num_qubits, [1], ["1" + "0"*(num_qubits-1)])
observable_two = hcp.generate_arbitary_observable(num_qubits, [1], ["2" + "0"*(num_qubits-1)])
observable_three = hcp.generate_arbitary_observable(num_qubits, [1], ["3" + "0"*(num_qubits-1)])
observables_list = [observable_one, observable_two, observable_three]
# dataforplot = [[],[],[],[],[],[]]
results = {}

statesinansatz = []

for g in g_vals:
    fidelity_results = dict()
    observable_expectation_results = dict()
    print(g)
    ############### Get Qutip Density Matrix
    hamiltonian = generate_fuji_boy_hamiltonian(num_qubits, g)
    gammas, L_terms = generate_fuji_boy_gamma_and_Lterms(num_qubits)
    qtp_hamiltonian = qutip.Qobj(hamiltonian.to_matrixform())
    qtp_Lterms = [qutip.Qobj(i.to_matrixform()) for i in L_terms]
    qtp_C_ops = [np.sqrt(gammas[i]) * qtp_Lterms[i] for i in range(len(qtp_Lterms))]
    qtp_rho_ss = qutip.steadystate(qtp_hamiltonian, qtp_C_ops,method="iterative-gmres",maxiter=10000)
    qtp_matrix = qtp_rho_ss.full()

    #compute the theoretical observable expectation values
    observable_matrixforms = [observable.to_matrixform() for observable in observables_list]
    theoretical_expectation_values = [np.trace(qtp_rho_ss.full() @ observable_matform) for observable_matform in observable_matrixforms]
    # dataforplot[0].append(theoretical_expectation_values[0])
    # dataforplot[1].append(theoretical_expectation_values[1])
    # dataforplot[2].append(theoretical_expectation_values[2])

    #################### Get the CsK states
    ansatz = acp.initial_ansatz(num_qubits)
    if what_starting_state == 'Ground_state':
        print('Using Ground state of TFI as initial state')
        start_state = qi.Statevector(np.array(qtp_hamiltonian.groundstate()[1]))
        # print(start_state)
        initial_state = acp.Initialstate(num_qubits, "starting_statevector", rand_generator=None,startingstatevector = start_state)

    elif what_starting_state == 'Random_statevector':
        print('Using Random Haar-random statevector as initial state')
        start_state = qi.random_statevector(2**num_qubits)
        # print(start_state)
        initial_state = acp.Initialstate(num_qubits, "starting_statevector", rand_generator=None,startingstatevector = start_state)

    for k in range(1, uptowhatK + 1):
        if random_selection_new:
            ansatz = acp.gen_next_ansatz(ansatz, hamiltonian, num_qubits,method='random_selection_new',num_new_to_add=numberofnewstatestoadd)
        else:
            ansatz = acp.gen_next_ansatz(ansatz, hamiltonian, num_qubits)

    ################ Create the new ansatz, only taking states below a support level
    ansatz = acp.remove_states_below_support(initial_state,ansatz,qtp_matrix,supportlevel=supportlevel)
    print('Number of states above support = %s'%(ansatz.get_ansatz_size()))
    statesinansatz.append(ansatz.get_ansatz_size())
    O_matrices_uneval = []
    for observable in observables_list:
        O_matrix_uneval = mcp.unevaluatedmatrix(num_qubits, ansatz, observable, "O")
        O_matrices_uneval.append(O_matrix_uneval)
    E_mat_uneval = mcp.unevaluatedmatrix(num_qubits, ansatz, hamiltonian, "E")
    D_mat_uneval = mcp.unevaluatedmatrix(num_qubits, ansatz, hamiltonian, "D")
    R_mats_uneval = []
    F_mats_uneval = []
    for thisL in L_terms:
        R_mats_uneval.append(mcp.unevaluatedmatrix(num_qubits,ansatz,thisL,"D"))
        thisLdagL = hcp.multiply_hamiltonians(hcp.dagger_hamiltonian(thisL),thisL)
        F_mats_uneval.append(mcp.unevaluatedmatrix(num_qubits,ansatz,thisLdagL,"D"))
    E_mat_evaluated = E_mat_uneval.evaluate_matrix_by_matrix_multiplicaton(initial_state)
    D_mat_evaluated = D_mat_uneval.evaluate_matrix_by_matrix_multiplicaton(initial_state)
    O_matrices_evaluated = [i.evaluate_matrix_by_matrix_multiplicaton(initial_state) for i in O_matrices_uneval]
    R_mats_evaluated = []
    for r in R_mats_uneval:
        R_mats_evaluated.append(r.evaluate_matrix_by_matrix_multiplicaton(initial_state))
    F_mats_evaluated = []
    for f in F_mats_uneval:
        F_mats_evaluated.append(f.evaluate_matrix_by_matrix_multiplicaton(initial_state))
    
    IQAE_instance = pp.IQAE_Lindblad(num_qubits, D_mat_evaluated, E_mat_evaluated,R_matrices = R_mats_evaluated,F_matrices = F_mats_evaluated,gammas = gammas)
    IQAE_instance.define_optimizer('feasibility_sdp', eigh_invcond=eigh_inv_cond,eig_invcond=eig_inv_cond,degeneracy_tol=degeneracy_tol,sdp_tolerance_bound=sdp_tolerance_bound)
    IQAE_instance.evaluate()
    result_dictionary = pp.analyze_density_matrix(num_qubits,initial_state,IQAE_instance,E_mat_evaluated,ansatz,hamiltonian,gammas,L_terms,qtp_rho_ss,O_matrices_evaluated)
    observable_expectation_results[1] = result_dictionary['observable_expectation']
    fidelity_results[1] = result_dictionary['fidelity']
    density_mat,groundstateenergy = IQAE_instance.get_density_matrix_results()
    results[g] = (observable_expectation_results, theoretical_expectation_values, fidelity_results,O_matrices_evaluated,density_mat)

theoretical_curves = plot_theoretical_expectation_curves(min(g_vals), max(g_vals), observables_list)
observable_names = [r'$<X_1>$',r'$<Y_1>$',r'$<Z_1>$']

plt.rcParams["figure.figsize"] = (7,5)
fidelity_plot_loc = 'reverse_engineer_ansatz_results/statesabovesupport%s_newgraph_%s_qubit_noiseless_fidelity.pdf'%(supportlevel,num_qubits)
# fidelity_plot_loc = None
plotp.plot_fidelities(num_qubits,results,False,None,x_axis=r'$g$',y_axis='Log(fidelity)', location=fidelity_plot_loc, bboxtight="tight",plotlog=True,k_dot_styles=["o","+","x","D","*","H"])

expectation_plot_loc = 'reverse_engineer_ansatz_results/statesabovesupport%s_newgraph_%s_qubit_noiseless.pdf'%(supportlevel,num_qubits)
# expectation_plot_loc = None
plotp.qutip_comparison_with_k_plot_expectation_values(num_qubits,results, theoretical_curves, [1],False,None,specify_names=True,observable_names=observable_names,x_axis=r'$g$',y_axis='Expectation Values', location=expectation_plot_loc, bboxtight="tight",k_dot_styles=["o","+","x","D","*","H"],line_styles=['-','--','-.'])

print("States in ansatz list")
print(statesinansatz)

    # dms = acp.calculate_ansatz_state_dms(initial_state,ansatz)
    # result = []
    # for dm in dms:
    #     result.append(np.trace(dm@qtp_matrix))
    # plt.plot(result)
    # plt.savefig('reverse_engineer_ansatz_results/supportoncsk_qubits%s_g%s.png'%(num_qubits,g))
    # plt.close()

    # fig, ax = plt.subplots()
    # img = ax.imshow(matrix_real,cmap='RdYlGn', interpolation='nearest')
    # clb = plt.colorbar(img)
    # # plt.show()
    # clb.ax.tick_params(labelsize=8) 
    # clb.ax.set_title('Real g=%s'%(g),fontsize=8)
    # plt.gca().set_aspect('equal', adjustable='box')
    # # plt.xlabel('h1')
    # # plt.xticks(np.linspace(0,matrix_real.shape[0]-1,11,endpoint=True),np.round(np.linspace(0,1.6,11,endpoint=True),3))
    # # plt.ylabel('%s'%(r'$\Omega$'))
    # # plt.ylabel('h2')
    # # plt.yticks(np.linspace(0,matrix_real.shape[0]-1,11,endpoint=True),np.round(np.linspace(-1.6,1.6,11,endpoint=True),3))
    # plt.savefig('reverse_engineer_ansatz_results/hintonreal_qubits%s_g%s.png'%(num_qubits,g))
    # plt.close()

    # fig, ax = plt.subplots()
    # img = ax.imshow(matrix_imag,cmap='RdYlGn', interpolation='nearest')
    # clb = plt.colorbar(img)
    # # plt.show()
    # clb.ax.tick_params(labelsize=8) 
    # clb.ax.set_title('Imag g=%s'%(g),fontsize=8)
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.savefig('reverse_engineer_ansatz_results/hintonimag_qubits%s_g%s.png'%(num_qubits,g))
    # plt.close()



