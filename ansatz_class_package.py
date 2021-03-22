import numpy as np
import hamiltonian_class_package as hcp
import pauli_class_package as pcp
from pauli_class_package import paulistring
from hamiltonian_class_package import Hamiltonian

#Qiskit functions. Since there is dependance on qiskit here, should make this explicit in README somehow
from qiskit.circuit.library import EfficientSU2 
from qiskit import execute, QuantumCircuit
from qiskit.providers.aer import Aer

class moment(object): #This moments are the building blocks of the Ansatz, basically its the moments that we used to build the chi states. This stores the alphas
    def __init__(self,N,paulistring,*alphas):# alphas is either nothing, or a list/numpy array
        self.N = N
        self.paulistring = paulistring#Should be paulistring class objects
        if len(alphas)==0:
            self.alphas = []
        else:
            self.alphas = alphas
    def get_paulistring(self):
        return paulistring
    
    def __repr__(self):
        """
        kh:
        I'm assuming that when we print a moment, we just print the string to help debug
        """
        return str(self.paulistring)

class Ansatz(object):#moments is a list
    def __init__(self,N,K,moments):
        self.N = N
        self.moments = moments#These are the moment objects
        self.K = K

    def get_moments(self):
        return self.moments
    
    def get_K(self):
        return self.K

    def get_alphas(self):
        resultpaulistrings = []
        resultalphas = []
        for mom in self.moments:
            resultpaulistrings.append(mom.paulistring.return_string())
            resultalphas.append(mom.alphas)
        return resultpaulistrings,resultalphas
    def __repr__(self):
        return str(self.moments)
        

class Initialstate(object):
    def __init__(self,N,method,numpyseed,numberoflayers):
        self.N = N
        self.method = method #Can be either random numbers or...
        self.numpyseed = numpyseed
        self.numberoflayers = numberoflayers
        self.qiskit_circuit = None

        if self.method == "efficient_SU2":
            qc = EfficientSU2(self.N, reps = self.numberoflayers, entanglement="full", skip_final_rotation_layer=False)
            num_params = qc.num_parameters 
            np.random.seed(self.numpyseed)
            initial_state_params = np.random.rand(num_params)
            for index in range(num_params):
                qc = qc.bind_parameters({qc.ordered_parameters[index]: initial_state_params[index]})
            self.qiskit_circuit = qc

    def get_statevector(self):
        if self.method == "random_numbers":
            dimension = 2**self.N 
            np.random.seed(self.numpyseed)
            state = np.random.rand(dimension) + 1j * np.random.rand(dimension)
            state = state / np.sqrt(np.vdot(state, state))
            return state 
        elif self.method == "efficient_SU2":
            statevector_backend = Aer.get_backend('statevector_simulator')
            state = execute(self.qiskit_circuit, statevector_backend).result().get_statevector()
            return state




    

def initial_ansatz(N):
    initialmoment = moment(N,paulistring(N,[0]*N,1))
    return Ansatz(N,0,[initialmoment])

def helper_get_strings(x):
    return x.return_string()

def gen_next_ansatz(anz,H,N,method = "no_processing"):
    if method == 'no_processing':
        newmomentstrings = []
        for mom in anz.moments:
            newmomentstrings.append(mom.return_string())

        for mom in anz.moments:
            for ham in H.return_paulistrings():
                newpauli = pcp.pauli_combine(mom.paulistring,ham)
                if newpauli.return_string() not in newmomentstrings:
                    newmomentstrings.append(newpauli.return_string())#This is the string that is [0,1,2,1,1,2,...] ect, NOT the paulistring class
        newmoment = []
        for i in newmomentstrings:
            newmoment.append(moment(N,paulistring(N,i,1)))#Appending the paulistring class objects
    return Ansatz(N,anz.K + 1,newmoment)

