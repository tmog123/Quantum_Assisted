import numpy as np
# import pauli_class_package as pcp
from qutip import *
import scipy as scp

number_qubits = 5
g_vals = [0.25]
howmanylargesteigenvectors = 5
uptowhatK = 2
randomselectionnumber = 5

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
    result = {}
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
            result[str(i)] = currentelements
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
            result[str(i)] = currentelements         
        print('Size of elements for %s is %s'%(i,len(currentelements)))
    return result



    


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
        if i in range(self.get_size):
            return self.statevectors[i]
        else:
            raise(RuntimeError("Not in range"))

thebasis = qutip_basis(number_qubits)

for g in g_vals:
    #Produces the steady state

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
    Observables = [thebasis['Xs'][0],thebasis['Ys'][0],thebasis['Zs'][0]]
    startingansatz = Ansatz(bigeigstatevectors)

    Csks = generate_CsKs_for_fuji_boy_hamiltonian(number_qubits,uptowhatK,thebasis,randomselectionnumber)







