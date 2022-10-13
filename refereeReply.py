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
import matplotlib.pyplot as plt



eigh_inv_cond = 10**(-6)
eig_inv_cond = 10**(-6)
degeneracy_tol = 5
loadmatlabmatrix = True
runSDPonpython = False

num_qubits = 8
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
# g = g_vals[4]

dataforplot = [[],[],[],[],[],[]]

observable_one = hcp.generate_arbitary_observable(num_qubits, [1], ["1" + "0"*(num_qubits-1)])
observable_two = hcp.generate_arbitary_observable(num_qubits, [1], ["2" + "0"*(num_qubits-1)])
observable_three = hcp.generate_arbitary_observable(num_qubits, [1], ["3" + "0"*(num_qubits-1)])
observable_obj_list = [observable_one, observable_two, observable_three]

for g in g_vals:
    print(g)

    hamiltonian = generate_fuji_boy_hamiltonian(num_qubits, g).to_matrixform()
    gammas, L_terms = generate_fuji_boy_gamma_and_Lterms(num_qubits)
    L_terms = [i.to_matrixform() for i in L_terms]
    L_dag_L_terms = [i.conj().T @ i for i in L_terms]

    #%%
    #get the steady state using qutip(lol)
    qtp_hamiltonian = qutip.Qobj(hamiltonian)
    qtp_Lterms = [qutip.Qobj(i) for i in L_terms]
    qtp_C_ops = [np.sqrt(gammas[i]) * qtp_Lterms[i] for i in range(len(qtp_Lterms))]
    qtp_rho_ss = qutip.steadystate(qtp_hamiltonian, qtp_C_ops, method="iterative-gmres",maxiter=10000)

    #compute the theoretical observable expectation values
    observable_matrixforms = [observable.to_matrixform() for observable in observable_obj_list]
    theoretical_expectation_values = [np.trace(qtp_rho_ss.full() @ observable_matform) for observable_matform in observable_matrixforms]
    # print(theoretical_expectation_values)
    dataforplot[0].append(theoretical_expectation_values[0])
    dataforplot[1].append(theoretical_expectation_values[1])
    dataforplot[2].append(theoretical_expectation_values[2])


    #%%
    # #doing the SDP
    # E_mat_evaluated = np.eye(len(hamiltonian[0]))
    # D_mat_evaluated = hamiltonian
    # R_mats_evaluated = L_terms
    # F_mats_evaluated = L_dag_L_terms

    # if loadmatlabmatrix == True:
    #     scipy.io.savemat("khstufftesting/Emat" +".mat",{"E": E_mat_evaluated,"D":D_mat_evaluated,"R":R_mats_evaluated,"F":F_mats_evaluated})
    #     print('Matrices have been generated, saved in khstufftestingfolder.')

    # #%%
    # if runSDPonpython == True:
    #     print("I'm here 1!")
    #     IQAE_instance = pp.IQAE_Lindblad(num_qubits, D_mat_evaluated, E_mat_evaluated,R_matrices = R_mats_evaluated,F_matrices = F_mats_evaluated,gammas = gammas)
    #     IQAE_instance.define_optimizer("feasibility_sdp", eigh_invcond=eigh_inv_cond,eig_invcond=eig_inv_cond,degeneracy_tol=degeneracy_tol,sdp_tolerance_bound=sdp_tolerance_bound)

    #     print("I'm here 2!")
    #     IQAE_instance.evaluate()
    #     density_mat,groundstateenergy = IQAE_instance.get_density_matrix_results()

    # elif loadmatlabmatrix == True:
    #     density_mat = scipy.io.loadmat('khstufftesting/'+'savedmatrixfrommatlab.mat')['betarho']

    density_mat = scipy.io.loadmat('khstufftesting/density_mat_g%s.mat'%(g))['denmat']

    sdp_expectation_values = [np.trace(density_mat @ observable_matform) for observable_matform in observable_matrixforms]
    dataforplot[3].append(sdp_expectation_values[0])
    dataforplot[4].append(sdp_expectation_values[1])
    dataforplot[5].append(sdp_expectation_values[2])

def plot_theoretical_expectation_curves(g_min,g_max, observable_obj_list):
    g_vals = np.linspace(g_min, g_max, 50)
    results = dict()
    for g in g_vals:
        hamiltonian = generate_fuji_boy_hamiltonian(num_qubits, g)
        gammas, L_terms = generate_fuji_boy_gamma_and_Lterms(num_qubits)
        qtp_hamiltonian = qutip.Qobj(hamiltonian.to_matrixform())
        qtp_Lterms = [qutip.Qobj(i.to_matrixform()) for i in L_terms]
        qtp_C_ops = [np.sqrt(gammas[i]) * qtp_Lterms[i] for i in range(len(qtp_Lterms))]
        qtp_rho_ss = qutip.steadystate(qtp_hamiltonian, qtp_C_ops,method="iterative-gmres",maxiter=10000)
        #compute the theoretical observable expectation values
        observable_matrixforms = [observable.to_matrixform() for observable in observable_obj_list]
        theoretical_expectation_values = [np.trace(qtp_rho_ss.full() @ observable_matform) for observable_matform in observable_matrixforms]
        results[g] = theoretical_expectation_values
    keys = list(results.keys())
    values = list(results.values())
    values_transposed = list(zip(*values)) 
    return (keys,values_transposed) #this is in a plottable form

theoretical_curves = plot_theoretical_expectation_curves(min(g_vals), max(g_vals), observable_obj_list)

plt.plot(theoretical_curves[0], theoretical_curves[1][0],linestyle='-', label = 'Theoretical'+r'$<X_1>$')
plt.plot(theoretical_curves[0], theoretical_curves[1][1],linestyle='--',label = 'Theoretical'+r'$<Y_1>$')
plt.plot(theoretical_curves[0], theoretical_curves[1][2],linestyle='-.',label = 'Theoretical'+r'$<Z_1>$')

# plt.plot(g_vals,np.real(dataforplot[0]),label='Theoretical'+r'$<X_1>$')
# plt.plot(g_vals,np.real(dataforplot[1]),label='Theoretical'+r'$<Y_1>$')
# plt.plot(g_vals,np.real(dataforplot[2]),label='Theoretical'+r'$<Z_1>$')
plt.plot(g_vals,dataforplot[3],"o",label='SDP'+r'$<X_1>$')
plt.plot(g_vals,dataforplot[4],"+",label='SDP'+r'$<Y_1>$')
plt.plot(g_vals,dataforplot[5],"x",label='SDP'+r'$<Z_1>$')
plt.tick_params(which='both', direction='in', top=True, right=True, labelsize=16)
plt.xlabel(r'$g$',fontsize=16)
plt.ylabel('Expectation Values',fontsize=16)
plt.legend()
plt.savefig('khstufftesting/8qubit.pdf',bbox_inches="tight")
    #%%
    # print(np.allclose(density_mat,qtp_rho_ss.full()))
    # %%
