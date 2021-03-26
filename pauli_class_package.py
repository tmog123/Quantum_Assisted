'''Pauli Package:
Deals with the functions that are needed to deal with the pauli strings

Functions to combine pauli strings, represent pauli strings

'''
import numpy as np

#Pauli matrices
identity = np.array([(1,0),(0,1)],dtype=np.complex128)
sigma_x = np.array([(0,1),(1,0)],dtype=np.complex128)
sigma_y = np.array([(0,-1j),(1j,0)],dtype=np.complex128)
sigma_z = np.array([(1,0),(0,-1)],dtype=np.complex128)
sigmas = [identity, sigma_x, sigma_y, sigma_z]


class paulistring(object):
    def __init__(self,N,string,coefficient):#String is represented as [0,2,2,1,3,...] where 0=I, 1=X, 2=Y, 3=Z
        if len(string) != N:
            raise(RuntimeError("N must match the length of string!"))
        self.N = N
        self.coefficient = coefficient
        self.string = string
    def return_string(self):
        return self.string
    def return_coefficient(self):
        return self.coefficient
    def __eq__(self, other):
        return self.coefficient == other.coefficient and self.string == other.string
    def __hash__(self):
        #first, we convert the internal list representation into a string so that it can be hashed
        #Tbh we most probably will not hash the pauli string object, but this is just in case
        stringified = "".join([str(i) for i in self.string])
        vals = [self.N, self.coefficient, stringified]
        keys = sorted(self.__dict__.items())
        return hash(tuple(zip(keys,vals)))
    def __str__(self):
        return self.__repr__()
    def get_complex_conjugate(self):
        return paulistring(self.N, self.string, np.conj(self.coefficient))
    def __repr__(self):
        stringified = "".join([str(i) for i in self.string])
        return str(self.coefficient) + "*"+stringified
    def get_matrixform(self):
        index_string = self.string
        coeff = self.coefficient
        first_matrix = sigmas[int(index_string[0])]
        matrix = first_matrix
        for j in index_string[1:]:
            matrix = np.kron(matrix, sigmas[int(j)])
        return coeff * matrix

def create_identity(N):
    return paulistring(N,N*[0],1)


def pauli_combine(pauli1,pauli2):
    resultN = pauli1.N
    resultcoeff = (pauli1.coefficient)*(pauli2.coefficient)
    resultstring = []
    for i in range(pauli1.N):
        subpauli,subcoeff = pauli_helper(pauli1.string[i],pauli2.string[i])
        #resultstring = resultstring + subpauli
        resultstring.append(subpauli)
        resultcoeff = resultcoeff*subcoeff
    return paulistring(resultN,resultstring,resultcoeff)


def pauli_helper(pauli1,pauli2):
    if pauli1 == 0:
        return pauli2,1
    if pauli2 == 0:
        return pauli1,1
    if pauli1 == 1:
        if pauli2 ==1:
            return 0,1
        elif pauli2 ==2:
            return 3,1j
        elif pauli2 ==3:
            return 2,-1j
    elif pauli1 == 2:
        if pauli2 ==1:
            return 3,-1j
        elif pauli2 ==2:
            return 0,1
        elif pauli2 ==3:
            return 1,1j
    elif pauli1 == 3:
        if pauli2 ==1:
            return 2,1j
        elif pauli2 ==2:
            return 1,-1j
        elif pauli2 ==3:
            return 0,1
























