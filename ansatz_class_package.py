import numpy as np
# import hamiltonian_class_package as hcp
import pauli_class_package as pcp
from pauli_class_package import paulistring
# from hamiltonian_class_package import Hamiltonian

#Qiskit functions. Since there is dependance on qiskit here, should make this explicit in README somehow
from qiskit.circuit.library import EfficientSU2 
from qiskit import execute, QuantumCircuit
from qiskit.providers.aer import Aer
from copy import deepcopy

class moment(object): #This moments are the building blocks of the Ansatz, basically its the moments that we used to build the chi states. This stores the alphas
    def __init__(self,N,paulistring,*alphas):# alphas is either nothing, or a list/numpy array
        self.N = N
        self.paulistring = paulistring#Should be paulistring class objects
        if len(alphas)==0:
            self.alphas = []
        else:
            self.alphas = alphas

    def get_paulistring(self):
        return self.paulistring
    
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
        self.initialalphassetyet = False

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

    def get_current_alphas(self):
        a,b = self.get_alphas()
        result = []
        for al in b:
            result.append(al[len(al)-1])
        return np.array(result)

    def update_alphas(self,newalphas):#the new alphas must be a list of lists
        for i in range(len(newalphas)):
            self.moments[i].alphas = newalphas[i]

    def __repr__(self):
        return str(self.moments)
        

class Initialstate(object):
    def __init__(self,N, method, rand_generator, numberoflayers = 2, qiskit_qc = None, startingstatevector = None):
        """
        method can be either efficient_SU2, or random_numbers, or...
        """
        self.N = N
        self.method = method #Can be either random numbers or...
        self.rand_generator = rand_generator
        #self.numpyseed = rand_generator
        self.numberoflayers = numberoflayers
        self.qiskit_circuit = None
        self.statevector_evaluatedbefore = False
        self.statevector = None

        if self.method == "own_qiskit_circuit" and qiskit_qc == None:
            raise(RuntimeError("You need to include a qiskit circuit to use this method"))

        if self.method == "starting_statevector":
            qc = QuantumCircuit(self.N)
            qc.set_statevector(startingstatevector)
            qc.save_state()
            self.qiskit_circuit = qc
            self.statevector_evaluatedbefore = True
            self.statevector = startingstatevector

        if self.method == "efficient_SU2":
            qc = EfficientSU2(self.N, reps = self.numberoflayers, entanglement="full", skip_final_rotation_layer=False)
            num_params = qc.num_parameters 
            #np.random.seed(self.numpyseed)
            #initial_state_params = np.random.rand(num_params)
            initial_state_params = self.rand_generator.random(num_params)
            for index in range(num_params):
                qc = qc.bind_parameters({qc.ordered_parameters[index]: initial_state_params[index]})
            self.qiskit_circuit = qc
        
        if self.method == "own_qiskit_circuit":
            self.qiskit_circuit = deepcopy(qiskit_qc)

        if self.method == "TFI_hardware_inspired":#Creates layers of single X rotations, followed by ZZ entangling gate rotations
            qc = QuantumCircuit(self.N)
            #np.random.seed(self.numpyseed)
            #self.startingrandomnumbers = np.random.rand(self.numberoflayers*(self.N + self.N-1)+self.N)
            self.startingrandomnumbers = self.rand_generator.random(self.numberoflayers*(self.N + self.N-1)+self.N)
            counter = 0
            for i in range(self.numberoflayers):
                for j in range(self.N):
                    qc.rx(self.startingrandomnumbers[counter],j)
                    counter = counter + 1
                for k in range(self.N-1):
                    qc.rzz(self.startingrandomnumbers[counter],k,k+1)
                    counter = counter + 1
            for j in range(self.N):
                qc.rx(self.startingrandomnumbers[counter],j)
                counter = counter + 1
            self.qiskit_circuit = qc

    def get_statevector(self):
        if self.method == "random_numbers" and self.statevector_evaluatedbefore==False:
            dimension = 2**self.N 
            #np.random.seed(self.numpyseed)
            state = self.rand_generator.random(dimension) + 1j * self.rand_generator.random(dimension)
            #state = np.random.rand(dimension) + 1j * np.random.rand(dimension)
            state = state / np.sqrt(np.vdot(state, state))
            self.statevector = state
            self.statevector_evaluatedbefore=True
            return state 
        elif self.method =="random_numbers" and self.statevector_evaluatedbefore==True:
            return self.statevector
        elif self.method == "efficient_SU2" or "own_qiskit_circuit":
            statevector_backend = Aer.get_backend('statevector_simulator')
            state = execute(self.qiskit_circuit, statevector_backend).result().get_statevector()
            return state
        elif self.method == "starting_statevector":
            return self.statevector
    
    def get_qiskit_circuit(self):
        if self.qiskit_circuit == None:
            #print('Method chosen has not been implemented in circuit fashion. Check ansatz_class_package for details')
            raise(RuntimeError('Method chosen has not been implemented in circuit fashion. Check ansatz_class_package for details'))
        else:
            return self.qiskit_circuit


def initial_ansatz(N):
    initialmoment = moment(N,paulistring(N,[0]*N,1))
    return Ansatz(N,0,[initialmoment])

def set_initial_ansatz_alpha_for_pruning(ansatz,num_steps):
    for i in range(num_steps):
        ansatz.get_moments()[0].alphas.append(1)


def helper_get_strings(x):
    return x.return_string()

def gen_next_ansatz(anz,H,N,method = "no_processing",pruning_condition = 0.1,num_new_to_add = 5):
    if method == 'no_processing':
        newmomentstrings = []
        for mom in anz.moments:
            newmomentstrings.append(mom.paulistring.return_string())
        for mom in anz.moments:
            for ham in H.return_paulistrings():
                newpauli = pcp.pauli_combine(mom.paulistring,ham)
                if newpauli.return_string() not in newmomentstrings:
                    newmomentstrings.append(newpauli.return_string())#This is the string that is [0,1,2,1,1,2,...] ect, NOT the paulistring class
        #print for debugging purposes
        print("there are " + str(len(newmomentstrings)) + " states in CSk")
        newmoment = []
        for i in newmomentstrings:
            newmoment.append(moment(N,paulistring(N,i,1)))#Appending the paulistring class objects
    if method == 'random_selection_new':
        oldmomentstrings = []
        for mom in anz.moments:
            oldmomentstrings.append(mom.paulistring.return_string())
        newmomentstrings = []
        for mom in anz.moments:
            for ham in H.return_paulistrings():
                newpauli = pcp.pauli_combine(mom.paulistring,ham)
                if newpauli.return_string() not in oldmomentstrings and newpauli.return_string() not in newmomentstrings:
                    newmomentstrings.append(newpauli.return_string())#This is the string that is [0,1,2,1,1,2,...] ect, NOT the paulistring class
        newmoment = []
        for i in oldmomentstrings:
            newmoment.append(moment(N,paulistring(N,i,1)))#Appending the paulistring class objects
        if len(newmomentstrings)<=num_new_to_add:
            for i in newmomentstrings:
                newmoment.append(moment(N,paulistring(N,i,1)))#Appending the paulistring class objects
        else:
            indicestoadd = np.random.choice(len(newmomentstrings),num_new_to_add,replace=False)
            for i in indicestoadd:
                newmoment.append(moment(N,paulistring(N,newmomentstrings[i],1)))#Appending the paulistring class objects
        print("there are " + str(len(newmoment)) + " states in CSk")
    if method == 'pruning':
        newmomentstrings = []
        for mom in anz.moments:
            maximum = max(mom.alphas)
            if maximum>pruning_condition:
                newmomentstrings.append(mom.paulistring.return_string())
        for mom in anz.moments:
            maximum = max(mom.alphas)
            if maximum>pruning_condition:
                for ham in H.return_paulistrings():
                    newpauli = pcp.pauli_combine(mom.paulistring,ham)
                    if newpauli.return_string() not in newmomentstrings:
                        newmomentstrings.append(newpauli.return_string())#This is the string that is [0,1,2,1,1,2,...] ect, NOT the paulistring class
        #print for debugging purposes
        print("there are " + str(len(newmomentstrings)) + " states in CSk")
        newmoment = []
        for i in newmomentstrings:
            newmoment.append(moment(N,paulistring(N,i,1)))#Appending the paulistring class objects
        
    return Ansatz(N,anz.K + 1,newmoment)

def set_initial_alphas(N,anz,method='start_with_initial_state'):
    if anz.initialalphassetyet == True:
        print('Initial alphas have already been set')
    else:
        print('Setting initial alphas')
        if method =='start_with_initial_state':
            identity = pcp.create_identity(N)
            for mom in anz.get_moments():
                if mom.paulistring.return_string() == identity.return_string():
                    mom.alphas.append(1)
                else:
                    mom.alphas.append(0)
        anz.initialalphassetyet = True