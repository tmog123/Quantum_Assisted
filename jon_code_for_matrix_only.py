import numpy as np
# import pauli_class_package as pcp
from qutip import *
import scipy as scp
import scipy.io
import copy
import post_processing as pp
import matplotlib.pyplot as plt
import plotting_package as plotp

number_qubits = 8
g_vals = [2.0,2.5,3.0]
howmanylargesteigenvectors = 20
uptowhatK = 1
randomselectionnumber = 10
eigh_inv_cond = 10**(-6)
eig_inv_cond = 10**(-6)
degeneracy_tol = 5
sdp_tolerance_bound = 0


def qutip_basis(L):
    correctannilations = enr_destroy([2]*L,excitations=L)
    correctidentity = enr_identity([2]*L,excitations=L)
    correctcreation = [c.dag() for c in correctannilations]
    # print(correctidentity)
    correctsigmazs = [(2*correctcreation[i]*correctannilations[i]-correctidentity) for i in range(L)]
    correctsigmays = [(-1j*(correctcreation[i]-correctannilations[i])) for i in range(L)]
    correctsigmaxs = [(correctcreation[i]+correctannilations[i]) for i in range(L)]
    # Xs = [x.data.toarray() for x in correctsigmaxs]
    # Ys = [y.data.toarray() for y in correctsigmays]
    # Zs = [z.data.toarray() for z in correctsigmazs]
    return {'Xs':correctsigmaxs,'Ys':correctsigmays,'Zs':correctsigmazs}

def generate_CsKs_for_fuji_boy_hamiltonian(num_qubits,uptowhatK,basis,howmanyrandomselection):
    result = {'0':[]}
    elements = []
    for i in range(num_qubits-1):
        elements.append(basis['Zs'][i]*basis['Zs'][i+1])
    for i in range(num_qubits):
        elements.append(g*basis['Xs'][i])
    currentelements = []
    for i in range(1,uptowhatK+1):
        if len(currentelements)==0:
            newelements = np.random.choice(len(elements),howmanyrandomselection,replace=False)
            for nw in newelements:
                currentelements.append(elements[nw])
        else:
            newelements = []
            for c in currentelements:
                for e in elements:
                    new = e*c
                    if (new not in currentelements) and (new not in newelements):
                        newelements.append(new)
            newelementstoadd = np.random.choice(len(newelements),howmanyrandomselection,replace=False)
            for nw in newelementstoadd:
                currentelements.append(newelements[nw])  
        result[str(i)] = copy.deepcopy(currentelements)
        print('Size of elements for %s is %s'%(i,len(currentelements)))
    return result

def evaluate_rho_dot(rho, qtp_hamiltonian, gammas, L_terms):
    hamiltonian_mat = qtp_hamiltonian.full()
    coherent_evo = -1j * (hamiltonian_mat @ rho - rho @ hamiltonian_mat)
    quantum_jumps_total = 0 + 0*1j
    for i in range(len(gammas)):
        gamma_i = gammas[i]
        L_i_mat = L_terms[i].full()
        L_i_dag_L_i = L_i_mat.conj().T @ L_i_mat
        anti_commutator = L_i_dag_L_i @ rho + rho @ L_i_dag_L_i
        jump_term = L_i_mat @ rho @ L_i_mat.conj().T
        quantum_jumps_total += gamma_i * (jump_term - 0.5*anti_commutator)
    return coherent_evo + quantum_jumps_total

def analyze_density_matrix(num_qubits,IQAE_instance,E_mat_evaluated,ansatz,qtp_hamiltonian,gammas,qtp_L_terms,qtp_rho_ss,O_matrices_evaluated, verbose = True):
    """
    hamiltonian, L_terms[i] are class objects, not matrices
    """
    density_mat,groundstateenergy = IQAE_instance.get_density_matrix_results()
    if type(density_mat) == type(None):
        print('SDP failed for this run, probably due to not high enough K')
    else:
        result_dictionary = {}
        IQAE_instance.check_if_valid_density_matrix()
        result_dictionary['trace_beta']=np.trace(density_mat @ E_mat_evaluated)
        result_dictionary['ground_state_energy']=groundstateenergy
        if verbose:
            print("the trace of the beta matrix is", np.trace(density_mat @ E_mat_evaluated))
            print('The ground state energy is\n',groundstateenergy)
        #print('The density matrix is\n',density_mat)
        result_dictionary["beta"] = density_mat
        if IQAE_instance.check_if_hermitian() == True:
            denmat_values,denmat_vects = scp.linalg.eigh(density_mat)
        else:
            denmat_values,denmat_vects = scp.linalg.eig(density_mat)
        denmat_values = np.real(np.round(denmat_values,10))
        #print(np.imag(denmat_values))
        if verbose:
            print("the sorted density matrix (beta matrix) eigenvalues are\n",np.sort(denmat_values))
        result_dictionary['sorted_beta_eigenvalues'] = np.sort(denmat_values)
        # p_string_matrices = [i.get_paulistring().get_matrixform() for i in ansatz.get_moments()]
        # ini_statevec_vecform = initial_state.get_statevector()
        # csk_states = [i@ini_statevec_vecform for i in p_string_matrices]
        rho = np.zeros(shape=(2**num_qubits,2**num_qubits), dtype = np.complex128)
        trace = 0
        for i in range(len(density_mat)):
            for j in range(len(density_mat)):
                i_j_entry = density_mat[(i,j)]
                i_j_ketbra = np.outer(ansatz.get_statevector(i), ansatz.get_statevector(j).conj().T)
                rho += i_j_entry * i_j_ketbra
                trace += i_j_entry * ansatz.get_statevector(j).conj().T @ ansatz.get_statevector(i)

        rho_eigvals,rho_eigvecs = scipy.linalg.eigh(rho)        
        result_dictionary['rho_eigvals'] = rho_eigvals
        result_dictionary['trace_rho'] = np.trace(rho)
        #now, we check if rho (the actual denmat) gives 0 for the linblad master equation
        #save the rho for processing
        result_dictionary["rho"] = rho
        rho_dot = evaluate_rho_dot(rho, qtp_hamiltonian, gammas, qtp_L_terms) #should be 0
        if verbose:
            print('rho_eigvals is: ' + str(rho_eigvals))
            print("trace rho is", np.trace(rho))
            print('Max value rho_dot is: ' + str(np.max(np.max(rho_dot))))
        result_dictionary['max_rho_dot'] = np.max(np.max(rho_dot))
        qtp_rho = qutip.Qobj(rho)
        # print(qtp_rho.data.shape)
        # print(qtp_rho_ss.data.shape)
        fidelity = qutip.metrics.fidelity(qutip.Qobj(qtp_rho.full()), qutip.Qobj(qtp_rho_ss.full()))
        if verbose:
            print("The fidelity is", fidelity)
        result_dictionary['fidelity'] = fidelity
        result_dictionary['observable_expectation'] = [np.trace(density_mat @ O_mat_eval) for O_mat_eval in O_matrices_evaluated]
        return result_dictionary

    


def generate_fuji_boy_hamiltonian(num_qubits, g,basis):
    H = 0
    for i in range(num_qubits-1):
        H = H - basis['Zs'][i]*basis['Zs'][i+1]
    for i in range(num_qubits):
        H = H - g*basis['Xs'][i]
    return H

def generate_fuji_boy_gamma_and_Lterms(num_qubits,basis):
    gammas_to_append = 1
    gammas = []
    L_terms = []
    for i in range(num_qubits):
        gammas.append(np.sqrt(gammas_to_append))
        L_terms.append(basis['Zs'][i])
        gammas.append(np.sqrt(gammas_to_append))
        L_terms.append(0.5*(basis['Xs'][i]-1j*basis['Ys'][i]))
    return (gammas, L_terms)

class Ansatz(object):#moments is a list
    def __init__(self,statevectors):
        self.statevectors = statevectors
    def get_size(self):
        return len(self.statevectors)
    def get_statevector(self,i):
        if i in range(self.get_size()):
            return self.statevectors[i]
        else:
            raise(RuntimeError("Not in range"))

def calculate_overlap(state1,state2):
    a = np.conjugate(np.transpose(state1))@state2
    return a

def produce_matrix(ansatz,qobj):
    if qobj == None:
        size = ansatz.get_size()
        matrix = np.empty([size,size], dtype=np.complex128)
        for i in range(size):
            for j in range(size):
                matrix[(i,j)] = calculate_overlap(ansatz.get_statevector(i),ansatz.get_statevector(j))
        return matrix
    else:
        size = ansatz.get_size()
        matrix = np.empty([size,size], dtype=np.complex128)
        H = qobj.full()
        for i in range(size):
            for j in range(size):
                matrix[(i,j)] = calculate_overlap(ansatz.get_statevector(i),H@ansatz.get_statevector(j))
        return matrix

def plot_theoretical_expectation_curves(g_min,g_max,num_qubits,basis, obs):
    g_vals = np.linspace(g_min, g_max, 30)
    results = dict()
    for g in g_vals:
        hamiltonian = generate_fuji_boy_hamiltonian(num_qubits, g,thebasis)
        gammas, L_terms = generate_fuji_boy_gamma_and_Lterms(num_qubits,thebasis)
        # qtp_hamiltonian = qutip.Qobj(hamiltonian.full())
        # qtp_Lterms = [qutip.Qobj(i.full()) for i in L_terms]
        qtp_C_ops = [np.sqrt(gammas[i]) * L_terms[i] for i in range(len(L_terms))]
        qtp_rho_ss = qutip.steadystate(hamiltonian, qtp_C_ops,method="iterative-bicgstab",maxiter=10000)
        #compute the theoretical observable expectation values
        observable_matrixforms = [ob.full() for ob in obs]
        theoretical_expectation_values = [np.trace(qtp_rho_ss.full() @ observable_matform) for observable_matform in observable_matrixforms]
        results[g] = theoretical_expectation_values
    keys = list(results.keys())
    values = list(results.values())
    values_transposed = list(zip(*values)) 
    return (keys,values_transposed) #this is in a plottable form

def numcskstatefunction(howmanyeigvec,randselect):
    def thefunc(k):
        return howmanyeigvec*(randselect*k) + howmanyeigvec
    return thefunc

thebasis = qutip_basis(number_qubits)
results = {}
Observables = [thebasis['Xs'][0],thebasis['Ys'][0],thebasis['Zs'][0]]
for g in g_vals:
    print('g is')
    print(g)
    #Produces the steady state
    fidelity_results = dict()
    observable_expectation_results = dict()
    qtp_hamiltonian = generate_fuji_boy_hamiltonian(number_qubits, g,thebasis)
    gammas, qtp_L_terms = generate_fuji_boy_gamma_and_Lterms(number_qubits,thebasis)
    if g == 0:
        qtp_rho_ss = qutip.steadystate(qtp_hamiltonian, qtp_L_terms, method="iterative-gmres",maxiter=10000)
    else:
        qtp_rho_ss = qutip.steadystate(qtp_hamiltonian, qtp_L_terms, method="iterative-bicgstab",maxiter=10000)
    qtp_matrix = qtp_rho_ss.full()
    print("theoretical steady state purity is", qtp_rho_ss.purity())
    bigeigvals,bigeigvecs = scp.sparse.linalg.eigsh(qtp_matrix,howmanylargesteigenvectors,which='LM')
    bigeigstatevectors = []
    for i in range(howmanylargesteigenvectors):
        bigeigstatevectors.append(np.transpose(np.conjugate(bigeigvecs[:,i])))
    # print(bigeigstatevectors[0].shape)

    #Start of algorithm
    theoretical_expectation_values = [np.trace(qtp_matrix @ ob.full()) for ob in Observables]
    # print(Observables[0])
    startingstatesansatz = Ansatz(bigeigstatevectors)


    # a = calculate_overlap(startingansatz.statevectors[0],startingansatz.statevectors[0])

    Csks = generate_CsKs_for_fuji_boy_hamiltonian(number_qubits,uptowhatK,thebasis,randomselectionnumber)

    # print(len(Csks['1']))

    for k in range(uptowhatK+1):
        thisstates = []
        for i in range(len(startingstatesansatz.statevectors)):
            thisstates.append(startingstatesansatz.get_statevector(i))
            for j in range(len(Csks[str(k)])):
                thisstates.append(Csks[str(k)][j].full()@startingstatesansatz.get_statevector(i))
        print('Size of ansatz for k = %s is %s'%(k,len(thisstates)))
        thiskansatz = Ansatz(thisstates)

        E_mat = produce_matrix(thiskansatz,None)
        D_mat = produce_matrix(thiskansatz,qtp_hamiltonian)
        O_mats = [produce_matrix(thiskansatz,i) for i in Observables]
        R_mats = []
        F_mats = []
        for L in qtp_L_terms:
            R_mats.append(produce_matrix(thiskansatz,L))
            F_mats.append(produce_matrix(thiskansatz,L.dag()*L))
        IQAE_instance = pp.IQAE_Lindblad(number_qubits, D_mat, E_mat,R_matrices = R_mats,F_matrices = F_mats,gammas = gammas)
        IQAE_instance.define_optimizer('feasibility_sdp', eigh_invcond=eigh_inv_cond,eig_invcond=eig_inv_cond,degeneracy_tol=degeneracy_tol,sdp_tolerance_bound=sdp_tolerance_bound)
        IQAE_instance.evaluate()
        result_dictionary = analyze_density_matrix(number_qubits,IQAE_instance,E_mat,thiskansatz,qtp_hamiltonian,gammas,qtp_L_terms,qtp_rho_ss,O_mats,verbose =False)
        observable_expectation_results[k] = result_dictionary['observable_expectation']
        fidelity_results[k] = result_dictionary['fidelity']
        print('Fidelity is ')
        print(result_dictionary['fidelity'])
        print('Observable Expectation is')
        print(result_dictionary['observable_expectation'])
    density_mat,groundstateenergy = IQAE_instance.get_density_matrix_results()
    results[g] = (observable_expectation_results, theoretical_expectation_values, fidelity_results,O_mats,density_mat)

numcskfunc = numcskstatefunction(howmanylargesteigenvectors,randomselectionnumber)

print('Stuck 1')
theoretical_curves = plot_theoretical_expectation_curves(min(g_vals), max(g_vals),number_qubits,thebasis, Observables)
observable_names = [r'$<X_1>$',r'$<Y_1>$',r'$<Z_1>$']
print('Stuck 2')
plt.rcParams["figure.figsize"] = (7,5)
fidelity_plot_loc = 'multiple_starting_state/startlargesteigvec_newgraph_%s_qubit_noiseless_fidelity.pdf'%(number_qubits)
plotp.plot_fidelities(number_qubits,results,True,numcskfunc,x_axis=r'$g$',y_axis='Log(fidelity)', location=fidelity_plot_loc, bboxtight="tight",plotlog=True,k_dot_styles=["o","+","x","D","*","H"])
print('Stuck 3')
expectation_plot_loc = 'multiple_starting_state/startlargesteigvec_newgraph_%s_qubit_noiseless.pdf'%(number_qubits)
plotp.qutip_comparison_with_k_plot_expectation_values(number_qubits,results, theoretical_curves, [0,1,2],True,numcskfunc,specify_names=True,observable_names=observable_names,x_axis=r'$g$',y_axis='Expectation Values', location=expectation_plot_loc, bboxtight="tight",k_dot_styles=["o","+","x","D","*","H"],line_styles=['-','--','-.'])



