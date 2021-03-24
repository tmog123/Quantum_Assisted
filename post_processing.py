import numpy as np 
import ansatz_class_package as acp 
import pauli_class_package as pcp 
import hamiltonian_class_package as hcp 
import matrix_class_package as mcp 
import scipy
import qutip as qt
import optimizers as opt_package

def helper_getcurrentalphas(alphas):
    result = []
    for al in alphas:
        result.append(al[len(al)-1])
    return np.array(result)

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
        self.eigh_invcond == eigh_invcond

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


class TTQS(object):
    def __init__(self,N,D_matrix,E_matrix,startingalphas):#These are the EVALUATED matrices
        self.N = N
        self.D = D_matrix
        self.E = E_matrix
        self.invcond = 10**(-6)
        self.startingalphas = startingalphas
        self.has_it_been_evaluated = False
    
    def numberstep(self,steps):
        self.steps = steps
    
    def define_endtime(self,endtime):
        self.endtime = endtime
    
    def define_optimizer(self,optimizer):
        self.optimizer = optimizer

    def define_backend(self,backend):
        self.backend = backend
    
    def define_invcond(self,ic):
        self.invcond = ic

    def evaluate(self):
        

        deltat = self.endtime/(self.steps-1)
        times=np.linspace(0,self.endtime,num=self.steps)
        self.G = self.E -1j*deltat*self.D
        alphas = self.startingalphas
        

        for t_idx,t in enumerate(times[:-1]):
            thisalpha = helper_getcurrentalphas(alphas)
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
    def get_results(self):
        if self.has_it_been_evaluated == False:
            print("This has not been evaluated yet")
        else:
            return self.finishedalphas








