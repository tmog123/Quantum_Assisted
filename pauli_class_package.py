'''Pauli Package:
Deals with the functions that are needed to deal with the pauli strings

Functions to combine pauli strings, represent pauli strings

'''
import numpy as np

class paulistring(object):
    def __init__(self,N,string,coefficient):#String is represented as [0,2,2,1,3,...] where 0=I, 1=X, 2=Y, 3=Z
        self.N = N
        self.coefficient = coefficient
        self.string = string
    def return_string(self):
        return self.string
    def return_coefficient(self):
        return self.coefficient

        
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
































