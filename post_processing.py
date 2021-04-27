import numpy as np 
import ansatz_class_package as acp 
import pauli_class_package as pcp 
import hamiltonian_class_package as hcp 
import matrix_class_package as mcp 
import scipy
import qutip as qt
import optimizers as opt_package
from abc import ABC, abstractmethod
from scipy.integrate import ode 

''' This is not necessary (But dont delete yet : Jonathan)
#This is stuff to get observables for classicalSimulator
X = np.array([[0.0, 1.0], [1.0, 0.0]])
Y = np.array([[0.0, -1j], [1j, 0.0]])
Z = np.array([[1.0, 0.0], [0.0, -1.0]])
P = np.array([[0.5, 0.5], [0.5, 0.5]])
II = np.array([[1.0, 0.0], [0.0, 1.0]])

# Function: Perform tensor product, output: I x I x I ....I x A x I
# A: An operator (e.g. Pauli operators)
# k: The position of particle
# N : Total number of particles
def Qtensor(A, k, N_total):
    return np.kron(np.identity(2**(k-1)), np.kron(A, np.identity(2**(N_total-k))))'''

def expectation_val(Qtensormatrix,psi):
    return psi.conj().dot(Qtensormatrix.dot(psi))

class classicalSimulator(object):
    """
    I.e, just does e^{-i H t}|\psi(\alpha(t = 0))>
    """
    def __init__(self,N,initialstate,hamiltonian,method='matrix_multiplication'):
        self.N = N
        self.initialstate = initialstate
        self.hamiltonian = hamiltonian
        self.times = None
        self.steps = None 
        self.endtime = None 
        self.results = None
        self.method = method
    def define_endtime(self,endtime):
        self.endtime = endtime
    def numberstep(self,steps):
        self.steps = steps
    
    def evaluate(self):
        if self.method == 'matrix_multiplication':
            startingstate = self.initialstate.get_statevector()
            self.times = np.linspace(0,self.endtime,num=self.steps)
            H = self.hamiltonian.to_matrixform()
            result = []
            for t in self.times:
                unitary = scipy.linalg.expm(-1j*t*H)
                evolvedstate = unitary@startingstate
                result.append(evolvedstate)
            self.results = result
    
    def get_results(self):#This returns a list of state vectors, the state vectors are the exact results for each timestep
        return self.results

    def get_expectations_observables(self,observablematrix):#This returns a list of the expecations values for each timestep
        expectations = []
        for vector in self.results:
            expectations.append(expectation_val(observablematrix,vector))
        return expectations



class IQAE(object):
    def __init__(self, N, D_matrix, E_matrix):
        """
        These are the evaluated matrices
        """
        self.N = N
        self.D = D_matrix
        self.E = E_matrix
        self.eigh_invcond = None
        self.optimizer = None
        self.ground_state_energy = None 
        self.ground_state_alphas = None
    
    def define_optimizer(self, optimizer, eigh_invcond = 10**(-12)):
        """
        Optimiser can either be eigh or qcqp or sdp or...
        Here we also take in the various invconds as arguments
        """
        self.optimizer = optimizer
        if optimizer == "eigh":
            self.set_eigh_invcond(eigh_invcond)
        elif optimizer == "qcqp":
            pass 
    
    def set_eigh_invcond(self, eigh_invcond):
        self.eigh_invcond = eigh_invcond

    def evaluate(self):
        if self.optimizer == None:
            raise(RuntimeError("run the define optimizer_function first"))

        elif self.optimizer == "eigh":
            qae_energies, qae_eigvecs = opt_package.diag_routine(self.D, self.E, inv_cond=self.eigh_invcond)
            min_energy = qae_energies[0]
            self.ground_state_energy = min_energy
            self.ground_state_alphas = qae_eigvecs[:,0]
    
    def get_results(self):
        return (self.ground_state_energy, self.ground_state_alphas)

#This is an abstract class
class quantumSimulators(ABC):
    @staticmethod
    def helper_getcurrentalphas(alphas):
        result = []
        for al in alphas:
            result.append(al[len(al)-1])
        return np.array(result)

    def __init__(self, N, D_matrix, E_matrix, startingalphas):
        """
        D_matrix and E_matrix here are evaluated matrices
        """
        self.N = N
        self.D = D_matrix
        self.E = E_matrix
        self.startingalphas = startingalphas
        self.steps = None 
        self.endtime = None 
        self.backend = None 
        self.has_it_been_evaluated = False
        self.finishedalphas = None

        #This was required in the past for debugging, but we don't need this now
        print('Ensuring E and D matrices are Hermitian by adding cc')
        self.E = (self.E + self.E.conj().T)/2
        self.D = (self.D + self.D.conj().T)/2

    def numberstep(self, steps):
        self.steps = steps 

    def define_endtime(self,endtime):
        self.endtime = endtime
    
    def define_backend(self,backend):
        self.backend = backend
    
    def get_results(self):
        if self.has_it_been_evaluated == False:
            print("This has not been evaluated yet")
        else:
            return self.finishedalphas

    @abstractmethod
    def define_optimizer(self,optimizer):
        pass

    @abstractmethod
    def evaluate(self):
        pass

class CQFF(quantumSimulators):
    def __init__(self, N, D_matrix, E_matrix, startingalphas, method = "diagonalise_H"):
        """
        Method can be either diagonalise_H (which is the main method in https://arxiv.org/pdf/2104.01931.pdf)

        or it can be U_dt (which is the proposed tweak in the discussion, in the same paper)
        """
        super().__init__(N, D_matrix, E_matrix, startingalphas)
        self.optimizer = None
        self.eigh_invcond = None
        self.times = None 
        self.method = method

    def define_optimizer(self, optimizer):
        self.optimizer = optimizer

    def define_eigh_invcond(self, eigh_invcond):
        self.eigh_invcond = eigh_invcond

    def get_times(self):
        return self.times 

    def evaluate(self):

        times=np.linspace(0,self.endtime,num=self.steps, endpoint=True)
        self.times = times
        alphas = self.startingalphas
        initial_alpha = super().helper_getcurrentalphas(alphas)

        if self.optimizer == "eigh":
            eigvals, eigvecs = opt_package.diag_routine(self.D, self.E, inv_cond=self.eigh_invcond)
            # print("eigvals are", eigvals)

            if self.method == "diagonalise_H":
                for t in times:
                    if t == 0:
                        continue
                    U_big_Delta_t_diagterms = np.exp(-1j*eigvals*t)
                    U_big_Delta_t = eigvecs @ np.diag(U_big_Delta_t_diagterms) @ eigvecs.conj().T @ self.E 
                    newalpha = U_big_Delta_t @ initial_alpha
                    for i in range(len(newalpha)):
                        alphas[i].append(newalpha[i])

            elif self.method == "U_dt":
                delta_t = times[1] - times[0]
                no_of_reps = len(times)
                for rep in range(1,no_of_reps + 1):
                    U_big_Delta_t_diagterms = (1 - 1j*eigvals*delta_t)**rep
                    U_big_Delta_t = eigvecs @ np.diag(U_big_Delta_t_diagterms) @ eigvecs.conj().T @ self.E 
                    newalpha = U_big_Delta_t @ initial_alpha
                    for i in range(len(newalpha)):
                        alphas[i].append(newalpha[i])

            self.has_it_been_evaluated = True
            self.finishedalphas = alphas


class QAS(quantumSimulators):
    def __init__(self, N, D_matrix, E_matrix, startingalphas):
        super().__init__(N, D_matrix, E_matrix, startingalphas)
        self.optimizer = None
        self.p_invcond = 10**(-6) #Default threshold for pseudo-inverse
        self.times = None

    def adot_vector(self, t, avec):
        return -1j*np.linalg.pinv(self.E, rcond = self.p_invcond)@self.D @ avec

    def define_optimizer(self, optimizer):
        self.optimizer = optimizer

    def define_p_invcond(self, p_invcond):
        self.p_invcond = p_invcond

    def get_times(self):
        return self.times 

    def evaluate(self):
        
        times=np.linspace(0,self.endtime,num=self.steps, endpoint=True)
        self.times = times
        alphas = self.startingalphas
        initial_alpha = super().helper_getcurrentalphas(alphas)
        solver = ode(self.adot_vector).set_integrator(self.optimizer)
        solver.set_initial_value(initial_alpha, 0)

        if self.optimizer == "zvode":
            previous_time = None 
            for t in times:
                if t == 0:
                    # newalpha = initial_alpha
                    previous_time = t
                    continue
                else:
                    time_advance_interval = t - previous_time 
                    previous_time = t 
                    newalpha = solver.integrate(solver.t + time_advance_interval)

                for i in range(len(newalpha)):
                    alphas[i].append(newalpha[i])

            self.has_it_been_evaluated = True
            self.finishedalphas = alphas

class TTQS(quantumSimulators):
    def __init__(self, N, D_matrix, E_matrix, startingalphas):
        super().__init__(N, D_matrix, E_matrix, startingalphas)
        self.optimizer = None
        self.invcond = 10**(-6) #Default
        self.times = None 

    def define_optimizer(self,optimizer):
        self.optimizer = optimizer

    def define_invcond(self,ic):
        self.invcond = ic

    def get_times(self):
        return self.times 

    def evaluate(self):

        deltat = self.endtime/(self.steps-1)
        times=np.linspace(0,self.endtime,num=self.steps)
        self.times = times[:] 
        self.G = self.E -1j*deltat*self.D
        alphas = self.startingalphas
        

        for t_idx,t in enumerate(times[:]): #why exclude the endpoint ah (Yeah, this was my mistake, should not be excluding endpoint: Jonathan)
            thisalpha = super().helper_getcurrentalphas(alphas)
            Wtop = np.outer(thisalpha,np.transpose(np.conjugate(thisalpha)))
            Wtop = np.matmul(self.G,Wtop)
            Wtop = np.matmul(Wtop,np.transpose(np.conjugate(self.G)))
            Wbot = np.matmul(np.transpose(np.conjugate(thisalpha)),self.E)
            Wbot = np.matmul(Wbot,thisalpha)
            W_matrix = Wtop/Wbot

            if self.optimizer == 'eigh':
                newalpha = opt_package.eigh_method_for_TTQS(self.E,W_matrix,thisalpha,self.invcond)
                for i in range(len(newalpha)):
                    alphas[i].append(newalpha[i])
        self.has_it_been_evaluated = True
        self.finishedalphas = alphas

# class TTQS(object):
#     def __init__(self,N,D_matrix,E_matrix,startingalphas):#These are the EVALUATED matrices
#         self.N = N
#         self.D = D_matrix
#         self.E = E_matrix
#         self.invcond = 10**(-6)
#         self.startingalphas = startingalphas
#         self.has_it_been_evaluated = False
    
#     def numberstep(self,steps):
#         self.steps = steps
    
#     def define_endtime(self,endtime):
#         self.endtime = endtime
    
#     def define_optimizer(self,optimizer):
#         self.optimizer = optimizer

#     def define_backend(self,backend):
#         self.backend = backend
    
#     def define_invcond(self,ic):
#         self.invcond = ic

#     def evaluate(self):
        

#         deltat = self.endtime/(self.steps-1)
#         times=np.linspace(0,self.endtime,num=self.steps)
#         self.G = self.E -1j*deltat*self.D
#         alphas = self.startingalphas
        

#         for t_idx,t in enumerate(times[:-1]):
#             thisalpha = helper_getcurrentalphas(alphas)
#             Wtop = np.outer(thisalpha,np.transpose(np.conjugate(thisalpha)))
#             Wtop = np.matmul(self.G,Wtop)
#             Wtop = np.matmul(Wtop,np.transpose(np.conjugate(self.G)))
#             Wbot = np.matmul(np.transpose(np.conjugate(thisalpha)),self.E)
#             Wbot = np.matmul(Wbot,thisalpha)
#             W_matrix = Wtop/Wbot

#             if self.optimizer == 'eigh':
#                 newalpha = opt_package.eigh_method_for_TTQS(self.E,W_matrix,thisalpha,self.invcond)
#                 for i in range(len(newalpha)):
#                     alphas[i].append(newalpha[i])
#         self.has_it_been_evaluated = True
#         self.finishedalphas = alphas

#     def get_results(self):
#         if self.has_it_been_evaluated == False:
#             print("This has not been evaluated yet")
#         else:
#             return self.finishedalphas