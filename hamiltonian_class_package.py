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

    def return_N(self):
        return self.N

    def to_matrixform(self):
        if len(self.paulistrings) < 1:
            raise(RuntimeError("Hamiltonian has no paulistrings yet"))
        first = self.paulistrings[0].get_matrixform()
        for j in range(1, len(self.paulistrings)):
            first += self.paulistrings[j].get_matrixform()
        return first 
    def return_betas(self):
        result = []
        for ps in self.paulistrings:
            result.append(ps.return_coefficient())
        return result


class Observable(Hamiltonian):
    def __init__(self, N, paulistrings):
        super().__init__(N, paulistrings)

def multiply_hamiltonians(ham1,ham2):#Multiplies 2 hamiltonians together and returns the new hamiltonian result
    if ham1.return_N() != ham2.return_N():
        raise(RuntimeError("Hamiltonians are not of same qubit size"))
    newpauliclassobjs = []
    stringsalready = []
    ham1pauliclassobjs = ham1.return_paulistrings()
    ham2pauliclassobjs = ham2.return_paulistrings()
    for p1 in ham1pauliclassobjs:
        for p2 in ham2pauliclassobjs:
            pnew = pcp.pauli_combine(p1,p2)
            #Avoid repeats of strings
            if pnew.return_string() in stringsalready:
                for pold in newpauliclassobjs:
                    if pnew.return_string() == pold.return_string():
                        pold.add_to_coefficient(pnew.return_coefficient())
            else:
                stringsalready.append(pnew.return_string())
                newpauliclassobjs.append(pnew)
    #create new hamiltonian
    return Hamiltonian(ham1.return_N(),newpauliclassobjs)

def dagger_hamiltonian(ham1):#Returns the dagger of the hamiltonian
    newpauliclassobjs = []
    N = ham1.return_N()
    for pold in ham1.return_paulistrings():
        newpauliclassobjs.append(pcp.paulistring(N,pold.return_string(),np.conj(pold.return_coefficient())))
    return Hamiltonian(N,newpauliclassobjs)



def generate_arbitary_hamiltonian(N, couplings, pauli_strings):
    """
    Here, N is the number of qubits
    Let the Hamiltonian be sum_{i=1}^r beta_i P_i
    Then, couplings = [beta_1, beta_2, ..., beta_L]
    pauli_strings = [P_1, P_2,..,P_L]
    Here, P_i is any iterable, e.g a string "123", or a list [1,2,3]. Both these iterables represent the operator X_1 Y_2 Z_3
    
    EXAMPLE: If we have a 4 qubit system and I want to implement the hamiltonian H = 0.6*(XXII) + 0.4*(XZIY), the hamiltonian will be generated with this:
    
    generate_arbitrary_hamiltonian(4,[0.6,0.4],["1100","1302"])
    
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

def transverse_ising_model_1d(N,J=1,g=1): #Sum -JZ_iZ_i+1 + Sum gX_i
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


def generate_package_of_random_hamiltonians(N,howmanyrandomhamiltonians,uptohowmanyterms,numpyseed,maximumbeta):#Returns a list of randomly generated hamiltonians
    random_generator = np.random.default_rng(numpyseed)
    ham_list = []
    for i in range(howmanyrandomhamiltonians):
        pauliterms = []
        while len(pauliterms)<uptohowmanyterms:
            rint = random_generator.integers(low=0,high=4,size=N)
            rstring = ''
            for j in rint:
                rstring = rstring+str(j)
            if rstring not in pauliterms:
                pauliterms.append(rstring)
        #print(pauliterms)
        paulistringobjects = []
        for term in pauliterms:
            paulistringobjects.append(pcp.paulistring(N,paulistringobjects,list(rint.random(uptohowmanyterms)*maximumbeta)))
        ham_list.append(Hamiltonian(N,paulistringobjects))
    return ham_list

