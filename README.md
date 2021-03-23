# Quantum-Assisted

Things done: Created the stuff needed to do simple IQAE, TTQS

Things to do: Create the code for normal QAS, create the code needed to generate the E and D matrices for IBM quantum computer backend, create the code for plotting and analyzing (right now the code only generates the alphas), create the code needed to do the exact calculations for QAS (probably using qutip) so that we can compare the results of TTQS/QAS with the expected results.

To see how it might work, can see the TTQS_tutorial.

Dependencies:

1. Qiskit
2. Numpy
3. Scipy

![image](https://user-images.githubusercontent.com/41319311/111591097-f44d8d00-8801-11eb-93a8-3804b6b6de88.png)

pauli_class_package:
1) Handles multiplication of pauli strings

Objects: Paulistring
Functions: pauli_combine, pauli_helper

hamiltonian_class_package:
1) Contains all the information of the Hamiltonian

REQUIRES PAULI CLASS

Objects: Hamiltonian
Functions: transverse_ising_model_1d, heisemberg_XYZ_model

ansatz_class_package:
1) Generates Ansatzes
2) Contains initial state clas representation

Objects: Ansatz, moment, Initialstate
Functions: initial_ansatz, gen_next_ansatz

matrix_class_package:
1) Contains the representations for the unevaluated E and D matrices (each element is a list of paulistring objects that will be passed to backend to evaluate)

post_processing:

1. The routine to solve the generalised eigenvalue problem
