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
num_qubits = 5
optimizer = 'feasibility_sdp'#'eigh' , 'eig', 'sdp','feasibility_sdp'
eigh_inv_cond = 10**(-6)
eig_inv_cond = 10**(-6)
degeneracy_tol = 5
use_qiskit = False
loadmatlabmatrix = False
runSDPonpython = True
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

for g in g_vals:
    print(g)
    ############### Get Qutip Density Matrix
    hamiltonian = generate_fuji_boy_hamiltonian(num_qubits, g)
    gammas, L_terms = generate_fuji_boy_gamma_and_Lterms(num_qubits)
    qtp_hamiltonian = qutip.Qobj(hamiltonian.to_matrixform())
    qtp_Lterms = [qutip.Qobj(i.to_matrixform()) for i in L_terms]
    qtp_C_ops = [np.sqrt(gammas[i]) * qtp_Lterms[i] for i in range(len(qtp_Lterms))]
    qtp_rho_ss = qutip.steadystate(qtp_hamiltonian, qtp_C_ops,method="iterative-gmres",maxiter=10000)
    # matrix = scp.sparse.csr_matrix(qtp_rho_ss.data)
    qtp_matrix = qtp_rho_ss.full()
    # print(type(matrix))

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
        print('##########################################')
        print('K = ' +str(k))
        # max_k_val = k
        #Generate Ansatz for this round
        if random_selection_new:
            ansatz = acp.gen_next_ansatz(ansatz, hamiltonian, num_qubits,method='random_selection_new',num_new_to_add=numberofnewstatestoadd)
        else:
            ansatz = acp.gen_next_ansatz(ansatz, hamiltonian, num_qubits)
    # print('Qutip matrix size shape is')
    # print(qtp_matrix.shape)
    dms = acp.calculate_ansatz_state_dms(initial_state,ansatz)
    result = []
    for dm in dms:
        result.append(np.trace(dm@qtp_matrix))
    plt.plot(result)
    plt.savefig('reverse_engineer_ansatz_results/supportoncsk_qubits%s_g%s.png'%(num_qubits,g))
    plt.close()

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



