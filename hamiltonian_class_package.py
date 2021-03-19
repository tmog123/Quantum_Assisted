import numpy as np
import pauli_class_package as pcp

class Hamiltonian(object):
    def __init__(self,N,paulistrings):
        self.N = N
        self.paulistrings = paulistrings#Should be paulistring class objects
    def __repr__(self):
        return str(self.paulistrings)
    def return_paulistrings(self):
        return self.paulistrings

def transverse_ising_model_1d(N,J,g): #Sum -JZ_iZ_i+1 + Sum gX_i
    paulistrings = []
    for i in range(N-1):
        st = [0]*N
        st[i]=3
        st[i+1]=3
        paulistrings.append(pcp.paulistring(N,st,-J))
    for i in range(N):
        st = [0]*N
        st[i]=1
        paulistrings.append(pcp.paulistring(N,st,g))
    return Hamiltonian(N,paulistrings)


