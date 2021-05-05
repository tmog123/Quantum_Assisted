# %%
# ### Import Packages
import numpy as np 
import ansatz_class_package as acp 
import pauli_class_package as pcp 
import hamiltonian_class_package as hcp 
import matrix_class_package as mcp 
import post_processing as pp
import plotting_package as plotp
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

num_qubits = 2
#a random matrix
linblad = np.array([np.random.rand(2**num_qubits) for i in range(2**num_qubits)])
ldagl = linblad.conj().T @ linblad

pauli_decomp = pcp.paulinomial_decomposition(ldagl) 

hamiltonian = hcp.generate_arbitary_hamiltonian(num_qubits, list(pauli_decomp.values()), list(pauli_decomp.keys()))

