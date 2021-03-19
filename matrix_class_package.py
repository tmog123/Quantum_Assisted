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
        
        if matrixtype == 'D':
            for i in range(size):
                for j in range(size):
                    element = []
                    for ham in H.return_paulistrings():
                        a = pcp.pauli_combine(moments[i].get_paulistring().get_complex_conjugate(),ham)
                        a = pcp.pauli_combine(a,moments[j].get_paulistring())
                        element.append(a)

                    numpymatrix[i,j] = element

        self.numpymatrix = numpymatrix




