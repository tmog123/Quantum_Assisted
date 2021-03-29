import numpy as np
import pauli_class_package as pcp

class Hamiltonian(object):
    def __init__(self,N,paulistrings):
        """
        N is the number of qubits 
        paulistrings is a list of paulistring class objects
        """
        self.N = N
        self.paulistrings = paulistrings#Should be paulistring class objects
    def __repr__(self):
        return str(self.paulistrings)

    def return_paulistrings(self):
        return self.paulistrings

    def to_matrixform(self):
        if len(self.paulistrings) < 1:
            raise(RuntimeError("Hamiltonian has no paulistrings yet"))
        first = self.paulistrings[0].get_matrixform()
        for j in range(1, len(self.paulistrings)):
            first += self.paulistrings[j].get_matrixform()
        return first 

class Observable(Hamiltonian):
    def __init__(self, N, paulistrings):
        super().__init__(N, paulistrings)

def generate_arbitary_hamiltonian(N, couplings, pauli_strings):
    """
    Here, N is the number of qubits
    Let the Hamiltonian be sum_{i=1}^r beta_i P_i
    Then, couplings = [beta_1, beta_2, ..., beta_L]
    pauli_strings = [P_1, P_2,..,P_L]
    Here, P_i is any iterable, e.g a string "123", or a list [1,2,3]. Both these iterables represent the operator X_1 Y_2 Z_3
    """
    if len(couplings) != len(pauli_strings):
        raise(RuntimeError("Length of couplings must match length of pauli_strings"))
    paulistring_objects = [] 
    for j in range(len(pauli_strings)):
        beta_j = couplings[j]
        P_j = pauli_strings[j]
        P_j_formatted = [int(i) for i in P_j]
        pauliobject = pcp.paulistring(N, P_j_formatted, beta_j)
        paulistring_objects.append(pauliobject)
    return Hamiltonian(N, paulistring_objects)

def generate_arbitary_observable(N, couplings, pauli_strings):
    """
    Here, N is the number of qubits
    Let the observable be sum_{i=1}^r beta_i P_i
    Then, couplings = [beta_1, beta_2, ..., beta_L]
    pauli_strings = [P_1, P_2,..,P_L]
    Here, P_i is any iterable, e.g a string "123", or a list [1,2,3]. Both these iterables represent the operator X_1 Y_2 Z_3
    """
    if len(couplings) != len(pauli_strings):
        raise(RuntimeError("Length of couplings must match length of pauli_strings"))
    paulistring_objects = [] 
    for j in range(len(pauli_strings)):
        beta_j = couplings[j]
        P_j = pauli_strings[j]
        P_j_formatted = [int(i) for i in P_j]
        pauliobject = pcp.paulistring(N, P_j_formatted, beta_j)
        paulistring_objects.append(pauliobject)
    return Observable(N, paulistring_objects)

def transverse_ising_model_1d(N,J,g): #Sum -JZ_iZ_i+1 + Sum gX_i
    """
    kh: what if N = 1?
    """
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

def heisenberg_xyz_model(N, jx = 1, jy = 2, jz = 3): #Sum jx X_i X_{i+1} + ...
    j_couplings = [jx,jy,jz]
    base_string = [0] * N 
    paulistrings = []
    if N == 1:
        for j in range(len(j_couplings)):
            if j_couplings[j] == 0:
                continue
            term = [j+1] 
            term = pcp.paulistring(N, term, j_couplings[j])
            paulistrings.append(term)
        return Hamiltonian(N, paulistrings)
    for i in range(N - 1):
        for j in range(len(j_couplings)):
            if j_couplings[j] == 0:
                continue
            term = base_string.copy() 
            term[i] = j + 1 
            term[i + 1] = j + 1
            term = pcp.paulistring(N, term, j_couplings[j])
            paulistrings.append(term)
    return Hamiltonian(N, paulistrings)