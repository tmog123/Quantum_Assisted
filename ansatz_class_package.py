import numpy as np
import hamiltonian_class_package as hcp
import pauli_class_package as pcp
from pauli_class_package import paulistring
from hamiltonian_class_package import Hamiltonian

class moment(object): #This moments are the building blocks of the Ansatz, basically its the moments that we used to build the chi states. This stores the alphas
    def __init__(self,N,paulistring,*alphas):# alphas is either nothing, or a list/numpy array
        self.N = N
        self.paulistring = paulistring#Should be paulistring class objects
        if len(alphas)==0:
            self.alphas = []
        else:
            self.alphas = alphas

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
        

def initial_ansatz(N):
    initialmoment = moment(N,paulistring(N,[0]*N,1))
    return Ansatz(N,0,[initialmoment])

def helper_get_strings(x):
    return x.return_string()

def gen_next_ansatz(anz,method,H,N):
    if method == 'no_processing':
        newmomentstrings = []
        for mom in anz.moments:
            for ham in H.return_paulistrings():
                newpauli = pcp.pauli_combine(mom.paulistring,ham.paulistring)
                if newpauli.return_string() not in newmomentstrings:
                    newmomentstrings.append(newpauli.return_string())#This is the string that is [0,1,2,1,1,2,...] ect, NOT the paulistring class
        newmoment = []
        for i in newmomentstrings:
            newmoment.append(moment(N,paulistring(N,i,1)))#Appending the paulistring class objects
    return Ansatz(N,anz.K + 1,newmoment)


