#%%
import numpy as np
import ansatz_class_package as acp 
import pauli_class_package as pcp 
import hamiltonian_class_package as hcp 
import matrix_class_package as mcp 
import post_processing as pp
import scipy as scp
import qutip
import math
import pandas as pd 

optimizer = 'feasibility_sdp'#'eigh' , 'eig', 'sdp','feasibility_sdp'
eigh_inv_cond = 10**(-6)
eig_inv_cond = 10**(-6)
use_qiskit = False
degeneracy_tol = 5

sdp_tolerance_bound = 0
howmanyrandominstances = 1

num_qubits = 4
uptowhatK = 2
which_hamiltonian = "sai_ring"
# which_hamiltonian = "bulk_dephasing"

random_selection_new = False
if random_selection_new == True:
    numberofnewstatestoadd = 100 #Only will be used if 'random_selection_new' is selected
    # numberofnewstatestoadd = 6 #Only will be used if 'random_selection_new' is selected

#Generate initial state
random_generator = np.random.default_rng(497)
initial_state = acp.Initialstate(num_qubits, "efficient_SU2", random_generator, 1)


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

def generate_bulk_dephasing(num_qubits):
    gammas = []
    L_terms = []
    if num_qubits == 1:
        raise(RuntimeError("One qubit case not considered"))
    else:
        for i in range(num_qubits):
            pauli_string_deconstructed = ["0"]*num_qubits
            pauli_string_deconstructed[i] = "3"
            pauli_string_str = "".join(pauli_string_deconstructed)
            L_i = hcp.generate_arbitary_hamiltonian(num_qubits, [1], [pauli_string_str])
            # print(L_i.to_matrixform())
            gammas.append(1)
            L_terms.append(L_i)
    return (gammas, L_terms)

def generate_total_magnetisation(num_qubits):
    def make_sigma_z_string(i):
        pauli_string_deconstructed = ["0"]*num_qubits
        pauli_string_deconstructed[i] = "3"
        pauli_string_str = "".join(pauli_string_deconstructed)
        return pauli_string_str
    p_strings = [make_sigma_z_string(i) for i in range(num_qubits)]
    betas = [1 for i in range(num_qubits)]
    M = hcp.generate_arbitary_hamiltonian(num_qubits, betas, p_strings)
    return M.to_matrixform()

def generate_parity_operator_matform(num_qubits):
    dim = 2**num_qubits
    P_mat = np.zeros((dim,dim))
    # test_mat = np.zeros(dim)
    for i in range(dim):
        ket_bitstring = np.binary_repr(i)
        ket_bitstring = (num_qubits - len(ket_bitstring))*"0" + ket_bitstring
        ket = np.zeros(dim)
        ket[i] = 1
        bra_bitstring = ket_bitstring[::-1]
        bra_index = int(bra_bitstring,2)
        # print(ket_bitstring, bra_bitstring)
        bra = np.zeros(dim)
        bra[bra_index] = 1
        P_mat += np.outer(ket,bra)
    spinflips = pcp.paulistring(num_qubits,[1]*num_qubits,1).get_matrixform()
    return P_mat @ spinflips

#%%

if which_hamiltonian == "sai_ring":
#Gamma and mu is for Sai ring model
    Gamma = 0.5
    mu = 0.4
    delta = 0.3

    #Sai ring hamiltonian
    hamiltonian = generate_XXZ_hamiltonian(num_qubits, delta)
    gammas, L_terms = generate_nonlocaljump_gamma_and_Lterms(num_qubits,Gamma,mu)


ansatz = acp.initial_ansatz(num_qubits)

#get the steady state using qutip(lol)
#This one is probably inaccurate!
qtp_hamiltonian = qutip.Qobj(hamiltonian.to_matrixform())
qtp_Lterms = [qutip.Qobj(i.to_matrixform()) for i in L_terms]
qtp_C_ops = [np.sqrt(gammas[i]) * qtp_Lterms[i] for i in range(len(qtp_Lterms))]
qtp_rho_ss = qutip.steadystate(qtp_hamiltonian, qtp_C_ops,method='iterative-gmres')
qtp_rho_ss_matform = qtp_rho_ss.full()

S_matform = generate_parity_operator_matform(num_qubits) 
S_matform_real = generate_parity_operator_matform(num_qubits) 

M_matform = generate_total_magnetisation(num_qubits) 
M_matform_real = generate_total_magnetisation(num_qubits) 

M_matform = M_matform - 4*np.eye(2**num_qubits) #pick out the eigenvalue I want, in this case its 4
M_paulis = pcp.paulinomial_decomposition(M_matform)
pauli_strings,couplings = list(zip(*M_paulis.items())) 
M = hcp.generate_arbitary_hamiltonian(num_qubits, couplings, pauli_strings) #in hcp.hamiltonian form

S_matform = S_matform - (1)*np.eye(2**num_qubits)
S_paulis = pcp.paulinomial_decomposition(S_matform)
pauli_strings,couplings = list(zip(*S_paulis.items())) 
S = hcp.generate_arbitary_hamiltonian(num_qubits, couplings, pauli_strings) #in hcp.hamiltonian form

# %%

fidelity_results = dict()

for k in range(uptowhatK):
    if random_selection_new:
        ansatz = acp.gen_next_ansatz(ansatz, hamiltonian, num_qubits,method='random_selection_new',num_new_to_add=numberofnewstatestoadd)
    else:
        ansatz = acp.gen_next_ansatz(ansatz, hamiltonian, num_qubits)


E_mat_uneval = mcp.unevaluatedmatrix(num_qubits, ansatz, hamiltonian, "E")
D_mat_uneval = mcp.unevaluatedmatrix(num_qubits, ansatz, hamiltonian, "D")
M_mat_uneval = mcp.unevaluatedmatrix(num_qubits, ansatz, M, "O")
S_mat_uneval = mcp.unevaluatedmatrix(num_qubits, ansatz, S, "O")

if optimizer == 'feasibility_sdp':
    R_mats_uneval = []
    F_mats_uneval = []
    for thisL in L_terms:
        R_mats_uneval.append(mcp.unevaluatedmatrix(num_qubits,ansatz,thisL,"D"))
        thisLdagL = hcp.multiply_hamiltonians(hcp.dagger_hamiltonian(thisL),thisL)
        F_mats_uneval.append(mcp.unevaluatedmatrix(num_qubits,ansatz,thisLdagL,"D"))


    E_mat_evaluated = E_mat_uneval.evaluate_matrix_by_matrix_multiplicaton(initial_state)
    D_mat_evaluated = D_mat_uneval.evaluate_matrix_by_matrix_multiplicaton(initial_state)
    M_tilde = M_mat_uneval.evaluate_matrix_by_matrix_multiplicaton(initial_state)
    S_tilde = S_mat_uneval.evaluate_matrix_by_matrix_multiplicaton(initial_state)

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

randombetainitializations = []
for i in range(howmanyrandominstances):
    randombetainitializations.append(random_generator.random((len(D_mat_evaluated),len(D_mat_evaluated))))
    # print(randombetainitializations[i])

results_dictionary = []

for betainitialpoint in randombetainitializations:

    if optimizer == 'feasibility_sdp':
        IQAE_instance = pp.IQAE_Lindblad(num_qubits, D_mat_evaluated, E_mat_evaluated,R_matrices = R_mats_evaluated,F_matrices = F_mats_evaluated,gammas = gammas)

    IQAE_instance.define_beta_initialpoint(betainitialpoint)
    IQAE_instance.define_optimizer(optimizer, eigh_invcond=eigh_inv_cond,eig_invcond=eig_inv_cond,degeneracy_tol=degeneracy_tol,sdp_tolerance_bound=sdp_tolerance_bound)
    IQAE_instance.define_additional_constraints_for_feasibility_sdp([[M_tilde,0], [S_tilde,0]])
    # IQAE_instance.define_additional_constraints_for_feasibility_sdp([[M_tilde,0]])
    IQAE_instance.evaluate()
# IQAE_instance.evaluate(kh_test=False)
#all_energies,all_states = IQAE_instance.get_results_all()
    results_dictionary.append(pp.analyze_density_matrix(num_qubits,initial_state,IQAE_instance,E_mat_evaluated,ansatz,hamiltonian,gammas,L_terms,qtp_rho_ss,[], verbose=False))
# observable_expectation_results[k] = result_dictionary['observable_expectation']
# fidelity_results[k] = result_dictionary['fidelity']

'''The results_dictionary is a list of all the result_dictionaries generated for each random beta initial point'''
def fidelity_checker(rho1, rho2):
    rho1 = rho1 / np.trace(rho1)
    rho2 = rho2 / np.trace(rho2)
    qtp_rho1 = qutip.Qobj(rho1)
    qtp_rho2 = qutip.Qobj(rho2)
    fidelity = qutip.metrics.fidelity(qtp_rho1, qtp_rho2)
    return fidelity

def handle_S_degeneracy(rho):
    rho_prime = S_matform @ rho @ S_matform.conjugate().transpose()
    rho_phys = 0.5*(rho + rho_prime) #works
    rho_phys = rho_phys / np.trace(rho_phys)
    print("fidelity between rho and rho phys is", fidelity_checker(rho, rho_phys))
    rhopp = (rho_phys + S_matform @ rho_phys)
    rhopp = rhopp / np.trace(rhopp)

    rhomm = (rho_phys - S_matform @ rho_phys)
    rhomm = rhomm / np.trace(rhomm)

    results = dict()
    results["rho_phys"] = rho_phys 
    results["rhopp"] = rhopp 
    results["rhomm"] = rhomm 
    return results 
#%%
result = results_dictionary[0] #since all the results are the same, just take the first one
rho = result['rho'] #should have tr(M_matform@rho) = 0 already
print(result["max_rho_dot"])
print(result["sorted_beta_eigenvalues"])
print(np.trace(rho@M_matform_real))
print(np.trace(rho@S_matform_real))

# rhoResults = handle_S_degeneracy(rho)
# rhopp = rhoResults["rhopp"]
# rhomm = rhoResults["rhomm"]


# rho_dot_pp = pp.evaluate_rho_dot(rhopp, hamiltonian,gammas, L_terms)
# print('Max value rho_dot_pp is: ' + str(np.max(np.max(rho_dot_pp))))

# rho_dot_mm = pp.evaluate_rho_dot(rhomm, hamiltonian,gammas, L_terms)
# print('Max value rho_dot_mm is: ' + str(np.max(np.max(rho_dot_mm))))

# print("tr(rhopp@S_matform), tr(rhopp@M)")
# print("Supposed to get S = 1, M = 0")
# print(np.trace(rhopp@S_matform), np.trace(rhopp@M_matform))
# print("tr(rhomm@S_matform), tr(rhomm@M)")
# print("Supposed to get S = -1, M = 0")
# print(np.trace(rhomm@S_matform), np.trace(rhomm@M_matform))
# # %%
