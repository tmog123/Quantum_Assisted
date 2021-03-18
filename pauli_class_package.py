'''Pauli Package:
Deals with the functions that are needed to deal with the pauli strings

Functions to combine pauli strings, represent pauli strings

'''
import numpy as np

class paulistring(object):
    def __init__(self,N,string,coefficient):
        self.N = N
        self.coefficient = coefficient
        self.string = string

        
def pauli_combine(pauli1,pauli2):
    resultN = pauli1.N
    resultcoeff = (pauli1.coefficient)*(pauli2.coefficient)
    resultstring = ''
    for i in range(pauli1.N):
        subpauli,subcoeff = pauli_helper(pauli1.string[i],pauli2.string[i])
        resultstring = resultstring + subpauli
        resultcoeff = resultcoeff*subcoeff
    return paulistring(resultN,resultstring,resultcoeff)


def pauli_helper(pauli1,pauli2):
    if pauli1 == 'I':
        return pauli2,1
    if pauli2 == 'I':
        return pauli1,1
    if pauli1 == 'X':
        if pauli2 =='X':
            return 'I',1
        elif pauli2 =='Y':
            return 'Z',1j
        elif pauli2 =='Z':
            return 'Y',-1j
    elif pauli1 == 'Y':
        if pauli2 =='X':
            return 'Z',-1j
        elif pauli2 =='Y':
            return 'I',1
        elif pauli2 =='Z':
            return 'X',1j
    elif pauli1 == 'Z':
        if pauli2 =='X':
            return 'Y',1j
        elif pauli2 =='Y':
            return 'X',-1j
        elif pauli2 =='Z':
            return 'I',1
































