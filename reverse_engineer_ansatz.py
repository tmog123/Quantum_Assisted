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

g_vals = [0,0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
num_qubits = 8

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
    hamiltonian = generate_fuji_boy_hamiltonian(num_qubits, g)
    gammas, L_terms = generate_fuji_boy_gamma_and_Lterms(num_qubits)
    qtp_hamiltonian = qutip.Qobj(hamiltonian.to_matrixform())
    qtp_Lterms = [qutip.Qobj(i.to_matrixform()) for i in L_terms]
    qtp_C_ops = [np.sqrt(gammas[i]) * qtp_Lterms[i] for i in range(len(qtp_Lterms))]
    qtp_rho_ss = qutip.steadystate(qtp_hamiltonian, qtp_C_ops,method="iterative-gmres",maxiter=10000)