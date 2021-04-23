import numpy as np
from ansatz_class_package import Ansatz
from ansatz_class_package import Initialstate
import pauli_class_package as pcp
import Qiskit_helperfunctions_Jon as Qhf

class unevaluatedmatrix(object):
    def __init__(self,N,ansatz,H_or_O,matrixtype):
        """
        Here, H_or_O is either a Hamiltonian class object or an Observable class object
        """
        self.N = N
        size = len(ansatz.get_moments())
        moments = ansatz.get_moments()
        dictionary_of_matrix_elements = dict()
        if matrixtype == 'E':
            # print("I'm here bij!")
            for i in range(size):
                for j in range(size):
                    element = [pcp.pauli_combine(moments[i].get_paulistring().get_complex_conjugate(),moments[j].get_paulistring())]

                    dictionary_of_matrix_elements[(i,j)] = element
        
        if matrixtype == 'D' or matrixtype == 'O': #actually the O matrix has the same form as the D matrix, just that the observable O takes the place of the Hamiltonian H
            for i in range(size):
                for j in range(size):
                    element = []
                    for ham in H_or_O.return_paulistrings():
                        a = pcp.pauli_combine(moments[i].get_paulistring().get_complex_conjugate(),ham)
                        a = pcp.pauli_combine(a,moments[j].get_paulistring())
                        element.append(a)

                    dictionary_of_matrix_elements[(i,j)] = element

        self.dict_of_uneval_matrix_elems = dictionary_of_matrix_elements
        self.size = size

    def evaluate_matrix_by_matrix_multiplicaton(self, initial_state_object):
        initial_statevector = initial_state_object.get_statevector()
        size = self.size
        matrix = np.empty([size,size], dtype=np.complex128)
        for i in range(size):
            for j in range(size):
                paulistrings = self.dict_of_uneval_matrix_elems[(i,j)]
                #by construction, paulistrings has at least 1 element
                temp_matrix = paulistrings[0].get_matrixform()
                #paulistring_object = paulistrings[0]
                #print(paulistring_object.get_string_for_hash())
                for k in range(1, len(paulistrings)):
                    #paulistring_object = paulistrings[k]
                    #print(paulistring_object.get_string_for_hash())
                    temp_matrix += paulistrings[k].get_matrixform()
                matrix[(i,j)] = initial_statevector.conj().T @ temp_matrix @ initial_statevector
        return matrix
    
    def return_set_of_pstrings_to_evaluate(self):
        """
        Takes the matrix and returns a list of the pstrings to be evaluated. This is a helper function to help prevent making duplicate measurements on the quantum computer.
        """
        to_return = set()
        for i in range(self.size):
            for j in range(self.size):
                p_string_str_forms = set([x.get_string_for_hash() for x in self.dict_of_uneval_matrix_elems[(i,j)]])
                to_return = to_return.union(p_string_str_forms)
        return to_return
    
    def substitute_evaluated_pstring_results(self, eval_results_dict):
        """
        Takes in a dictionary, with the key value pairs being:
        key -- tensored-Pauli operator P
        value -- <psi|P|psi>, evaluated either by classical means or by the quantum computer

        Then, returns an evaluated matrix
        """
        size = self.size
        matrix = np.empty([size,size], dtype=np.complex128)
        for i in range(size):
            for j in range(size):
                paulistrings = self.dict_of_uneval_matrix_elems[(i,j)]
                #by construction, paulistrings has at least 1 element
                value = 0 + 0j
                for p in paulistrings:
                    result = eval_results_dict[p.get_string_for_hash()]
                    value += result * p.return_coefficient()
                matrix[(i,j)] = value
        return matrix

    def evaluate_matrix_with_qiskit_circuits(self,initial_state_object,sim='noiseless',shots=8192,whichcomputer=None,noisebackend=None):
        print('Evaluating matrix with Qiskit Circuits')
        #initial_qiskitcircuit = initial_state_object.get_qiskit_circuit()
        size = self.size
        matrix = np.empty([size,size], dtype=np.complex128)
        #pastresultsevaluated = {}
        for i in range(size):
            for j in range(size):
                paulistrings = self.dict_of_uneval_matrix_elems[(i,j)]
                #by construction, paulistrings has at least 1 element
                temporary = 0
                for k in range(len(paulistrings)):
                
                    #thispaulistring = paulistrings[k].get_string_for_hash()
                    coeff = paulistrings[k].return_coefficient()

                    paulistring_object = paulistrings[k]
                    #print(paulistring_object.get_string_for_hash())
                    a = Qhf.evaluate_circuit(self.N,initial_state_object,paulistring_object,sim,shots=shots,whichrealcomputer=whichcomputer,noisebackend=noisebackend)
                    temporary = temporary + a*coeff
                    #if thispaulistring in pastresultsevaluated.keys():
                    #    temporary = temporary + pastresultsevaluated[thispaulistring]*coeff
                    #else:
                    #    paulistring_object = paulistrings[k]
                    #    a = Qhf.evaluate_circuit(self.N,initial_state_object,paulistring_object,sim,shots)
                    #    temporary = temporary + a*coeff
                    #    pastresultsevaluated[thispaulistring] = a
                matrix[(i,j)] = temporary
        return matrix

def evaluate_pstrings_strings_classicaly(set_of_pstrings_strforms, initial_statevector):
    ans = dict()
    for pstring_strform in set_of_pstrings_strforms:
        pstring = pcp.paulistring(len(pstring_strform), pstring_strform, 1)
        pstring_matform = pstring.get_matrixform() 
        ans[pstring_strform] = initial_statevector.conj().T @ pstring_matform @ initial_statevector
    return ans

