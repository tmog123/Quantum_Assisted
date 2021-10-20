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

uptowhatK = 3
sdp_tolerance_bound = 0
num_qubits = 4
howmanyrandominstances = 1

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
    sim = "noiseless_qasm"
    quantum_com = "ibmq_bogota" #which quantum computer to take the noise profile from
    num_shots = 30000 #max is 1000000

    # sim = "noisy_qasm"
    # quantum_com = "ibmq_bogota" #which quantum computer to take the noise profile from
    # num_shots = 8192 #max is 8192

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

#%%
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

def generate_total_magnetisation_matform(num_qubits):
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

Gamma = 0.5
mu = 0.4
delta = 0.3


hamiltonian = generate_XXZ_hamiltonian(num_qubits, delta)
gammas, L_terms = generate_nonlocaljump_gamma_and_Lterms(num_qubits,Gamma,mu)
ansatz = acp.initial_ansatz(num_qubits)

# ansatz = acp.gen_next_ansatz(ansatz, hamiltonian, num_qubits,method='random_selection_new',num_new_to_add=numberofnewstatestoadd)

#%%
#get the steady state using qutip(lol)
#This one is probably inaccurate!
qtp_hamiltonian = qutip.Qobj(hamiltonian.to_matrixform())
qtp_Lterms = [qutip.Qobj(i.to_matrixform()) for i in L_terms]
qtp_C_ops = [np.sqrt(gammas[i]) * qtp_Lterms[i] for i in range(len(qtp_Lterms))]
qtp_rho_ss = qutip.steadystate(qtp_hamiltonian, qtp_C_ops,method='iterative-gmres')
qtp_rho_ss_matform = qtp_rho_ss.full()

S = generate_parity_operator_matform(num_qubits)
M = generate_total_magnetisation_matform(num_qubits)
S_m = scp.linalg.expm(1j*M*np.pi/(num_qubits*2))
# %%

fidelity_results = dict()

uptowhatK = 2
for k in range(uptowhatK):
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
    else:
        IQAE_instance = pp.IQAE_Lindblad(num_qubits, D_mat_evaluated, E_mat_evaluated)

    IQAE_instance.define_beta_initialpoint(betainitialpoint)
    IQAE_instance.define_optimizer(optimizer, eigh_invcond=eigh_inv_cond,eig_invcond=eig_inv_cond,degeneracy_tol=degeneracy_tol,sdp_tolerance_bound=sdp_tolerance_bound)

    IQAE_instance.evaluate()
# IQAE_instance.evaluate(kh_test=False)
#all_energies,all_states = IQAE_instance.get_results_all()
    results_dictionary.append(pp.analyze_density_matrix(num_qubits,initial_state,IQAE_instance,E_mat_evaluated,ansatz,hamiltonian,gammas,L_terms,qtp_rho_ss,[], verbose=False))
# observable_expectation_results[k] = result_dictionary['observable_expectation']
# fidelity_results[k] = result_dictionary['fidelity']

'''The results_dictionary is a list of all the result_dictionaries generated for each random beta initial point'''

'''COMMENTED OUT THE BELOW: JON'''

result = results_dictionary[0] #since all the results are the same, just take the first one
rho = result['rho']

#%%
# Here we apply the algorithm for multiple NESS for the case of S_m, since there is a degeneracy due to S_m
#For this case, we are considering the case of even N in order to get the eigenvalue corresponding to M = 0 (i.e, S_m = 1). In this subspace, we expect rho_pp and rho_mm. 
# For our case, for even N, the number of eigenvalues of S_m is N+1. We can take this into consideration in our algorithm. 
# Hence, for N = 4, we have 5*5 possible NESS, 5 of which are legit. Hence in general, for the front part of our algo, we need to run it 5*5 - 5 times to eliminate the 20 non-physical NESS
#First we handle the S_m degeneracy to get the steady state with zero magnetisation, then we handle the S degeneracy to get rho_pp and rho_ss in the zero magnetisation subspace

S_m_eigvals = np.array(list(set(scp.linalg.eigvals(S_m)))) #set because unique eigvals only

def handle_S_m_degeneracy(rhoStart):
    # rhoStart = rhopp
    """
    returns the steady state such that tr(rho S_m) = 1, i.e returns the steady state with 0 magnetisation
    """
    rhoTemp = rhoStart
    for i in range(num_qubits):
        for j in range(num_qubits):
            if i == j:
                continue 
            else:
                rhoTempTemp = rhoTemp - S_m @ rhoStart @ S_m.conj().T 
                factor = 1 - np.exp(1j*(S_m_eigvals[i] - S_m_eigvals[j]))
                rhoTemp = rhoTemp - (1/factor) * rhoTempTemp

    rhoPhys = rhoTemp
    vanderMat = np.vander(S_m_eigvals, increasing = True).T
    vanderMatInv = scp.linalg.inv(vanderMat)

    matrices = []
    for i in range(len(S_m_eigvals)):
        factor = np.linalg.matrix_power(S_m, i)
        matrices.append(factor @ rhoPhys)
    matrices = np.array(matrices) 

    index = 1 #index corresponding to M = 0
    vanderMatVec = vanderMatInv[index]
    last = vanderMatVec[-1] * matrices[-1]
    for j in range(len(matrices) - 1):
        last += vanderMatVec[j] * matrices[j] 

    last = last / np.trace(last)

    last_dot = pp.evaluate_rho_dot(last, hamiltonian, gammas, L_terms)
    print(np.max(np.max(last_dot)))

    return last 

def handle_S_degeneracy(rho):
    rho_prime = S @ rho @ S.conjugate().transpose()
    rho_phys = 0.5*(rho + rho_prime) #works
    rho_phys = rho_phys / np.trace(rho_phys)

    rhopp = (rho_phys + S @ rho_phys)
    rhopp = rhopp / np.trace(rhopp)

    rhomm = (rho_phys - S @ rho_phys)
    rhomm = rhomm / np.trace(rhomm)

    results = dict()
    results["rho_phys"] = rho_phys 
    results["rhopp"] = rhopp 
    results["rhomm"] = rhomm 
    return results 

def fidelity_checker(rho1, rho2):
    rho1 = rho1 / np.trace(rho1)
    rho2 = rho2 / np.trace(rho2)
    qtp_rho1 = qutip.Qobj(rho1)
    qtp_rho2 = qutip.Qobj(rho2)
    fidelity = qutip.metrics.fidelity(qtp_rho1, qtp_rho2)
    return fidelity

rhotst = handle_S_m_degeneracy(rho)
rhoResults = handle_S_degeneracy(rhotst)
rhopp = rhoResults["rhopp"]
rhomm = rhoResults["rhomm"]


# qtp_rhotst = handle_S_m_degeneracy(qtp_rho_ss_matform) #this does nothing!
# since qtp_rho_ss_matform is already in the 4 magnetisation subspace
qtp_rhoResults = handle_S_degeneracy(qtp_rho_ss_matform)
# qtp_rhoResults = handle_S_degeneracy(qtp_rhotst)
qtp_rhopp = qtp_rhoResults["rhopp"]
qtp_rhomm = qtp_rhoResults["rhomm"]
qtp_rhophys = qtp_rhoResults["rho_phys"]


#%% test
rhoStart = qtp_rhomm
rhoTemp = rhoStart
for i in range(num_qubits):
    for j in range(num_qubits):
        if i == j:
            continue 
        else:
            rhoTempTemp = rhoTemp - S_m @ rhoStart @ S_m.conj().T 
            factor = 1 - np.exp(1j*(S_m_eigvals[i] - S_m_eigvals[j]))
            rhoTemp = rhoTemp - (1/factor) * rhoTempTemp

rhoPhys = rhoTemp
vanderMat = np.vander(S_m_eigvals, increasing = True).T
vanderMatInv = scp.linalg.inv(vanderMat)

matrices = []
for i in range(len(S_m_eigvals)):
    factor = np.linalg.matrix_power(S_m, i)
    matrices.append(factor @ rhoPhys)
matrices = np.array(matrices) 

index = 3 
vanderMatVec = vanderMatInv[index]
last = vanderMatVec[-1] * matrices[-1]
for j in range(len(matrices) - 1):
    last += vanderMatVec[j] * matrices[j] 

display(pd.DataFrame(last))
print(np.trace(last))
last = last / np.trace(last)

# last_dot = pp.evaluate_rho_dot(last, hamiltonian, gammas, L_terms)
# print(np.max(np.max(last_dot)))

#%%
rho_dot_pp = pp.evaluate_rho_dot(rhopp, hamiltonian,gammas, L_terms)
print('Max value rho_dot_pp is: ' + str(np.max(np.max(rho_dot_pp))))

rho_dot_mm = pp.evaluate_rho_dot(rhomm, hamiltonian,gammas, L_terms)
print('Max value rho_dot_mm is: ' + str(np.max(np.max(rho_dot_mm))))

qtp_rho_dot_pp = pp.evaluate_rho_dot(qtp_rhopp, hamiltonian,gammas, L_terms)
print('Max value qtp_rho_dot_pp is: ' + str(np.max(np.max(qtp_rho_dot_pp))))

qtp_rho_dot_mm = pp.evaluate_rho_dot(qtp_rhomm, hamiltonian,gammas, L_terms)
print('Max value qtp_rho_dot_mm is: ' + str(np.max(np.max(qtp_rho_dot_pp))))

rhopp_fidelity = fidelity_checker(qtp_rhopp, rhopp) 
rhomm_fidelity = fidelity_checker(qtp_rhomm, rhomm) 

print("rhopp_fidelity, rhomm_fidelity is", rhopp_fidelity,rhomm_fidelity)