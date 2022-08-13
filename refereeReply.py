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



eigh_inv_cond = 10**(-6)
eig_inv_cond = 10**(-6)
degeneracy_tol = 5
loadmatlabmatrix = False
runSDPonpython = True

num_qubits = 7
# uptowhatK = 1
sdp_tolerance_bound = 0

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




g_vals = [0,0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
g = g_vals[4]

observable_one = hcp.generate_arbitary_observable(num_qubits, [1], ["1" + "0"*(num_qubits-1)])
observable_two = hcp.generate_arbitary_observable(num_qubits, [1], ["2" + "0"*(num_qubits-1)])
observable_three = hcp.generate_arbitary_observable(num_qubits, [1], ["3" + "0"*(num_qubits-1)])
observable_obj_list = [observable_one, observable_two, observable_three]

hamiltonian = generate_fuji_boy_hamiltonian(num_qubits, g).to_matrixform()
gammas, L_terms = generate_fuji_boy_gamma_and_Lterms(num_qubits)
L_terms = [i.to_matrixform() for i in L_terms]
L_dag_L_terms = [i.conj().T @ i for i in L_terms]

#%%
#get the steady state using qutip(lol)
qtp_hamiltonian = qutip.Qobj(hamiltonian)
qtp_Lterms = [qutip.Qobj(i) for i in L_terms]
qtp_C_ops = [np.sqrt(gammas[i]) * qtp_Lterms[i] for i in range(len(qtp_Lterms))]
qtp_rho_ss = qutip.steadystate(qtp_hamiltonian, qtp_C_ops, method="iterative-gmres")

#compute the theoretical observable expectation values
observable_matrixforms = [observable.to_matrixform() for observable in observable_obj_list]
theoretical_expectation_values = [np.trace(qtp_rho_ss.full() @ observable_matform) for observable_matform in observable_matrixforms]


#%%
#doing the SDP
E_mat_evaluated = np.eye(len(hamiltonian[0]))
D_mat_evaluated = hamiltonian
R_mats_evaluated = L_terms
F_mats_evaluated = L_dag_L_terms

print("I'm here 1!")
IQAE_instance = pp.IQAE_Lindblad(num_qubits, D_mat_evaluated, E_mat_evaluated,R_matrices = R_mats_evaluated,F_matrices = F_mats_evaluated,gammas = gammas)
IQAE_instance.define_optimizer("feasibility_sdp", eigh_invcond=eigh_inv_cond,eig_invcond=eig_inv_cond,degeneracy_tol=degeneracy_tol,sdp_tolerance_bound=sdp_tolerance_bound)

print("I'm here 2!")
IQAE_instance.evaluate()
density_mat,groundstateenergy = IQAE_instance.get_density_matrix_results()

#%%
print(np.allclose(density_mat,qtp_rho_ss.full()))
# %%
