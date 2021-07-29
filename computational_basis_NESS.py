#%%
import numpy as np 
import scipy as scp
import qutip 
# import ansatz_class_package as acp 
# import pauli_class_package as pcp 
import hamiltonian_class_package as hcp 
# import matrix_class_package as mcp 
# import post_processing as pp
# import scipy.io
import optimizers as opt_package

import random
import pandas as pd 
from functools import partial 
from tqdm.notebook import tqdm
from pqdm.threads import pqdm
import pickle
import os
import matplotlib.pyplot as plt

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

def generate_submatrix(numpy_matrix, indices):
    final_ans = np.empty((len(indices), len(indices)))
    final_ans = final_ans + 1j*final_ans
    for i in range(len(indices)):
        for j in range(len(indices)):
            row_index = indices[i]
            col_index = indices[j]
            final_ans[(i,j)] = numpy_matrix[(row_index, col_index)]
    return final_ans

def submatrix_to_full_matrix(num_qubits, submatrix, indices):
    hilbert_space_dimension = 2**num_qubits
    final_ans = np.zeros((hilbert_space_dimension, hilbert_space_dimension))
    final_ans = final_ans + 1j*final_ans
    for i in range(len(indices)):
        for j in range(len(indices)):
            row_index = indices[i]
            col_index = indices[j]
            final_ans[(row_index,col_index)] = submatrix[(i,j)]
    return final_ans

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
#%% Start of code
num_qubits = 5
sai_or_fuji = "sai"

if sai_or_fuji == "sai":
    Gamma = 0.9
    mu = 0.5

def big_loop(num_qubits, num_states, g):
    hilbert_space_dimension = 2**num_qubits
    # num_states = hilbert_space_dimension // 2
    random_indices = sorted(random.sample(range(hilbert_space_dimension),num_states))

    # print("The random indices are", random_indices)

    # observable_one = hcp.generate_arbitary_observable(num_qubits, [1], ["1" + "0"*(num_qubits-1)])
    # observable_two = hcp.generate_arbitary_observable(num_qubits, [1], ["2" + "0"*(num_qubits-1)])
    # observable_three = hcp.generate_arbitary_observable(num_qubits, [1], ["3" + "0"*(num_qubits-1)])
    # observables_list = [observable_one, observable_two, observable_three]

    if sai_or_fuji == "fuji":
        hamiltonian = generate_fuji_boy_hamiltonian(num_qubits,g)
        gammas,L_terms = generate_fuji_boy_gamma_and_Lterms(num_qubits)
    elif sai_or_fuji == "sai":
        hamiltonian = generate_XXZ_hamiltonian(num_qubits, g)
        gammas, L_terms = generate_nonlocaljump_gamma_and_Lterms(num_qubits, Gamma, mu)

    hamiltonian_matform = hamiltonian.to_matrixform()
    L_terms_matform = [i.to_matrixform() for i in L_terms]

    qtp_hamiltonian = qutip.Qobj(hamiltonian_matform)
    qtp_Lterms = [qutip.Qobj(i) for i in L_terms_matform]
    qtp_C_ops = [np.sqrt(gammas[i]) * qtp_Lterms[i] for i in range(len(qtp_Lterms))]
    qtp_rho_ss = qutip.steadystate(qtp_hamiltonian, qtp_C_ops, method = "svd")

    #generate matrices
    E_matrix = generate_submatrix(np.eye(hilbert_space_dimension), random_indices)
    D_matrix = generate_submatrix(hamiltonian_matform, random_indices)
    R_matrices = [generate_submatrix(i, random_indices) for i in L_terms_matform]
    F_matrices = [generate_submatrix(i.conj().T @ i, random_indices) for i in L_terms_matform]

    beta = opt_package.cvxpy_density_matrix_feasibility_sdp_routine(D_matrix,E_matrix,R_matrices,F_matrices,gammas,0, verbose = False)[0]

    rho = submatrix_to_full_matrix(num_qubits, beta, random_indices)

    rho_dot = evaluate_rho_dot(rho, hamiltonian, gammas, L_terms) #should be 0
    print('Max value rho_dot is: ' + str(np.max(np.max(rho_dot))))
    qtp_rho = qutip.Qobj(rho)
    fidelity = qutip.metrics.fidelity(qtp_rho, qtp_rho_ss)
    # print("The fidelity is", fidelity)
    return fidelity



hilbert_space_dimension = 2**num_qubits

if sai_or_fuji == "fuji":
    g_vals = [0, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
elif sai_or_fuji == "sai":
    delta_vals = [0,0.125,0.25,0.375, 0.5,0.625,0.75,0.875, 1.0, 1.5, 2.0]
    g_vals = delta_vals #renaming stuff for consistency below

num_states_list = [hilbert_space_dimension // i for i in (8,4,2,1)]
num_states_list.append(26)
num_states_list.sort()

def inner_loop(num_states, num_runs = 30):
    big_loop_curried = partial(big_loop, num_qubits, num_states)
    avg_fidelities = np.zeros(len(g_vals))
    for run in tqdm(range(num_runs), leave = False):
        fidelities = np.array(pqdm(g_vals, big_loop_curried, n_jobs = len(g_vals), leave = False))
        avg_fidelities += fidelities 
    avg_fidelities /= num_runs
    return avg_fidelities

def main():
    results_all_k = dict()
    for num_states in tqdm(num_states_list, leave = False):
        avg_fidelities = inner_loop(num_states)
        results = {i:j for (i,j) in zip(g_vals, avg_fidelities)}
        results_all_k[num_states] = results
    if sai_or_fuji == "fuji":
        save_obj(results_all_k, "computational_basis_ness_fujiBoy_5_qubits")
    elif sai_or_fuji == "sai":
        save_obj(results_all_k, "computational_basis_ness_sai_5_qubits")

if sai_or_fuji == "fuji":
    # main()
    results_all_k = load_obj("computational_basis_ness_fujiBoy_5_qubits")
else:
    main()
    results_all_k = load_obj("computational_basis_ness_sai_5_qubits")

# num_states = 26
# avg_fidelities = inner_loop(num_states)
# results = {i:j for (i,j) in zip(g_vals, avg_fidelities)}
# results_all_k[num_states] = results
# save_obj(results_all_k, "computational_basis_ness_5_qubits")


def plot_fidelities(results,savefile = None):
    num_states_list = list(results_all_k.keys())
    num_states_list.sort()
    for index in range(len(num_states_list)):
        num_states = num_states_list[index]
        data = results[num_states]
        x_vals = list(data.keys())
        y_vals = list(data.values())
        plt.plot(x_vals,y_vals, label="num states="+str(num_states))

    plt.xlabel("g")
    plt.ylabel("fidelity")
    plt.title(str(num_qubits) + " qubits fidelity graph")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    if savefile:
        plt.savefig(savefile,bbox_inches="tight")
        plt.close()
    else:
        plt.show()

plot_fidelities(results_all_k)
# %%
