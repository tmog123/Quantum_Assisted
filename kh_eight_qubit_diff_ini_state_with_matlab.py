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
import pickle
import os
from tqdm.notebook import tqdm


def save_obj(obj, dir_name, name):
    #name is a string
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    with open(dir_name +'/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(dir_name, name):
    #name is a string
    with open(dir_name +'/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


eigh_inv_cond = 10**(-6)
eig_inv_cond = 10**(-6)
degeneracy_tol = 5
loadmatlabmatrix = True
runSDPonpython = False
num_qubits = 8
sdp_tolerance_bound = 0

uptowhatK = 4
random_selection_new = True
if random_selection_new == True:
    numberofnewstatestoadd = 50 #Only will be used if 'random_selection_new' is selected

what_starting_state = 'largest_eigvec'# 'efficient_SU2_Random', 'Ground_state', 'Random_statevector', 'largest_eigvec'
if what_starting_state == 'efficient_SU2_Random':
    random_generator = np.random.default_rng(123)
    initial_state = acp.Initialstate(num_qubits, "efficient_SU2", random_generator, 1)

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


def generate_all_matrices_and_csk(g):
    print("\n")
    print("g is", g)
    hamiltonian = generate_fuji_boy_hamiltonian(num_qubits, g)
    hamiltonian_matform = hamiltonian.to_matrixform()
    gammas, L_terms = generate_fuji_boy_gamma_and_Lterms(num_qubits)
    L_terms_matform = [i.to_matrixform() for i in L_terms]
    # L_dag_L_terms_matform = [i.conj().T @ i for i in L_terms]

    #get the steady state using qutip(lol)
    qtp_hamiltonian = qutip.Qobj(hamiltonian_matform)
    qtp_Lterms = [qutip.Qobj(i) for i in L_terms_matform]
    qtp_C_ops = [np.sqrt(gammas[i]) * qtp_Lterms[i] for i in range(len(qtp_Lterms))]
    if g == 0:
        qtp_rho_ss = qutip.steadystate(qtp_hamiltonian, qtp_C_ops, method="iterative-gmres",maxiter=10000)
    else:
        qtp_rho_ss = qutip.steadystate(qtp_hamiltonian, qtp_C_ops, method="iterative-bicgstab",maxiter=10000)
    print("theoretical steady state purity is", qtp_rho_ss.purity())

    #start of our algorithm
    ansatz = acp.initial_ansatz(num_qubits)
    if what_starting_state == 'Ground_state':
        print('Using Ground state of TFI as initial state')
        start_state = qi.Statevector(np.array(qtp_hamiltonian.groundstate()[1]))
        # print(start_state)
        initial_state = acp.Initialstate(num_qubits, "starting_statevector", rand_generator=None,startingstatevector = start_state)

    elif what_starting_state == 'largest_eigvec':
        qtp_matrix = qtp_rho_ss.data
        bigeigval,bigeigvec = scp.sparse.linalg.eigsh(qtp_matrix,1,which='LM')
        bigeigvec = np.array(bigeigvec)
        bigeigvec = bigeigvec/np.sqrt(np.vdot(bigeigvec, bigeigvec))
        overlapWithNESS = np.vdot(bigeigvec, qtp_rho_ss.full() @ bigeigvec)
        print("starting state overlap amplitude with ness is", overlapWithNESS)
        start_state = qi.Statevector(bigeigvec)
        initial_state = acp.Initialstate(num_qubits, "starting_statevector", rand_generator=None,startingstatevector = start_state)

    elif what_starting_state == 'Random_statevector':
        print('Using Random Haar-random statevector as initial state')
        start_state = qi.random_statevector(2**num_qubits)
        # print(start_state)
        initial_state = acp.Initialstate(num_qubits, "starting_statevector", rand_generator=None,startingstatevector = start_state)

    for k in tqdm(range(1, uptowhatK + 1)):
        try:
            if random_selection_new:
                ansatz = acp.gen_next_ansatz(ansatz, hamiltonian, num_qubits,method='random_selection_new',num_new_to_add=numberofnewstatestoadd)
                print('Ansatz Size is %s'%(ansatz.get_ansatz_size()))

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
            R_mats_evaluated = []
            for r in R_mats_uneval:
                R_mats_evaluated.append(r.evaluate_matrix_by_matrix_multiplicaton(initial_state))
            F_mats_evaluated = []
            for f in F_mats_uneval:
                F_mats_evaluated.append(f.evaluate_matrix_by_matrix_multiplicaton(initial_state))

            p_string_matrices = [i.get_paulistring().get_matrixform() for i in ansatz.get_moments()]
            ini_statevec_vecform = initial_state.get_statevector()
            csk_states = [i@ini_statevec_vecform for i in p_string_matrices]

            scipy.io.savemat("8_qubit_EmatsAndCsk/Emat_g={}_noOfStates={}".format(g, len(csk_states)) +".mat",{"E": E_mat_evaluated,"D":D_mat_evaluated,"R":R_mats_evaluated,"F":F_mats_evaluated})

            save_obj(csk_states, "8_qubit_EmatsAndCsk", "cskStates_g={}_noOfStates={}".format(g, len(csk_states)))
        except:
            continue

# Code to generate the E matrices
# g_vals = [0,0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
# for g in tqdm(g_vals):
#     generate_all_matrices_and_csk(g)


# Code to generate all the rho
# base_dir = "8_qubit_EmatsAndCsk" 
# csk_dir = "allCSKstates_smaller"
# betarho_dir = "allbetarho_smaller"

# all_csk_fnames = sorted(os.listdir(base_dir+"/" + csk_dir))
# all_betarho_fnames = sorted(os.listdir(base_dir+"/" + betarho_dir))

# for m in tqdm(range(len(all_csk_fnames))):
#     csk_states = load_obj(base_dir+"/"+csk_dir, all_csk_fnames[m][:-4])
#     beta_rho_matrix = scipy.io.loadmat(base_dir + "/"+betarho_dir+"/"+all_betarho_fnames[m])["betarho"]
#     rho = np.zeros(shape=(2**num_qubits,2**num_qubits), dtype = np.complex128)
#     density_mat = beta_rho_matrix
#     # trace = 0
#     for i in range(len(density_mat)):
#         for j in range(len(density_mat)):
#             i_j_entry = density_mat[(i,j)]
#             i_j_ketbra = np.outer(csk_states[i], csk_states[j].conj().T)
#             rho += i_j_entry * i_j_ketbra
#     save_obj(rho, base_dir+"/"+"allrho_smaller", "rho_for_"+all_csk_fnames[m][:-4])

#Code to do what we want 
base_dir = "8_qubit_EmatsAndCsk"
rho_dir = "allrho_smaller"
all_rho_dir_fnames = os.listdir(base_dir+"/" + rho_dir)
rho_dict = dict() #in terms of g
for dirname in all_rho_dir_fnames:
    g_val = float(dirname[-4:])
    rho_dict[g_val] = []
    for fname in os.listdir(base_dir+"/" + rho_dir + "/" + dirname):
        numCsk = fname[-7:-4]
        if "=" in numCsk:
            numCsk = numCsk[1:]
        numCsk = int(numCsk)
        rho = load_obj(base_dir+"/" + rho_dir + "/" + dirname, fname[:-4])
        rho_dict[g_val].append((numCsk, rho))

for g in rho_dict.keys():
    print("g is", g)
    hamiltonian = generate_fuji_boy_hamiltonian(num_qubits, g)
    hamiltonian_matform = hamiltonian.to_matrixform()
    gammas, L_terms = generate_fuji_boy_gamma_and_Lterms(num_qubits)
    L_terms_matform = [i.to_matrixform() for i in L_terms]
    # L_dag_L_terms_matform = [i.conj().T @ i for i in L_terms]

    #get the steady state using qutip(lol)
    qtp_hamiltonian = qutip.Qobj(hamiltonian_matform)
    qtp_Lterms = [qutip.Qobj(i) for i in L_terms_matform]
    qtp_C_ops = [np.sqrt(gammas[i]) * qtp_Lterms[i] for i in range(len(qtp_Lterms))]
    if g == 0:
        qtp_rho_ss = qutip.steadystate(qtp_hamiltonian, qtp_C_ops, method="iterative-gmres",maxiter=10000)
    else:
        qtp_rho_ss = qutip.steadystate(qtp_hamiltonian, qtp_C_ops, method="iterative-bicgstab",maxiter=10000)
    print("theoretical steady state purity is", qtp_rho_ss.purity())

    #start of our algorithm
    ansatz = acp.initial_ansatz(num_qubits)
    if what_starting_state == 'Ground_state':
        print('Using Ground state of TFI as initial state')
        start_state = qi.Statevector(np.array(qtp_hamiltonian.groundstate()[1]))
        # print(start_state)
        initial_state = acp.Initialstate(num_qubits, "starting_statevector", rand_generator=None,startingstatevector = start_state)

    elif what_starting_state == 'largest_eigvec':
        qtp_matrix = qtp_rho_ss.data
        bigeigval,bigeigvec = scp.sparse.linalg.eigsh(qtp_matrix,1,which='LM')
        bigeigvec = np.array(bigeigvec)
        bigeigvec = bigeigvec/np.sqrt(np.vdot(bigeigvec, bigeigvec))
        overlapWithNESS = np.vdot(bigeigvec, qtp_rho_ss.full() @ bigeigvec)
        print("starting state overlap amplitude with ness is", overlapWithNESS)
        # start_state = qi.Statevector(bigeigvec)
        # initial_state = acp.Initialstate(num_qubits, "starting_statevector", rand_generator=None,startingstatevector = start_state)

    print("\n")
    for (numCsk,rho) in rho_dict[g]:
        try:
            qtp_rho = qutip.Qobj(rho)
            fidelity = qutip.metrics.fidelity(qtp_rho, qtp_rho_ss)
            print("We have {} CsK states in our ansatz".format(numCsk))
            print("Fidelity is", fidelity)
            print("\n")
        except ValueError as e: 
            print("Ran into a value error when we used {} CsK states".format(numCsk)) 
            print("\n")
            continue