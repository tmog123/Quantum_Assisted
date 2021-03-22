import numpy as np
from ansatz_class_package import Ansatz
from ansatz_class_package import Initialstate
import pauli_class_package as pcp


class unevaluatedmatrix(object):
    def __init__(self,N,ansatz,H,matrixtype):
        self.N = N
        size = len(ansatz.get_moments())
        moments = ansatz.get_moments()
        numpymatrix = np.zeros([size,size])
        if matrixtype == 'E':
            for i in range(size):
                for j in range(size):
                    element = [pcp.pauli_combine(moments[i].get_paulistring().get_complex_conjugate(),moments[j].get_paulistring())]
                    numpymatrix[i,j] = element
        
        if matrixtype == 'D' or 'O': #actually the O matrix has the same form as the D matrix, just that the observable O takes the place of the Hamiltonian H
            for i in range(size):
                for j in range(size):
                    element = []
                    for ham in H.return_paulistrings():
                        a = pcp.pauli_combine(moments[i].get_paulistring().get_complex_conjugate(),ham)
                        a = pcp.pauli_combine(a,moments[j].get_paulistring())
                        element.append(a)

                    numpymatrix[i,j] = element

        self.numpymatrix = numpymatrix
    
    def evaluate_matrix_by_matrix_multiplicaton(self, initial_state_object):
        initial_statevector = initial_state_object.get_statevector()
        size = len(self.numpymatrix[0])
        matrix = np.empty([size,size], dtype=np.complex128)
        for i in range(size):
            for j in range(size):
                paulistrings = self.numpymatrix[(i,j)]
                #by construction, paulistrings has at least 1 element
                temp_matrix = paulistrings[0].get_matrixform()
                for k in range(1, len(paulistrings)):
                    temp_matrix += paulistrings[k].get_matrixform()
                matrix[(i,j)] = initial_statevector.conj().T @ temp_matrix @ initial_statevector
        return matrix