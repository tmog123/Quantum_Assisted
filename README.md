# Quantum-Assisted

Things done: Created the classes to represent the Ansatz and Hamiltonians, with associating functions to generate them

Things to do: Create the code needed to generate the E and D matrices (will probably be backend dependent), create the code for the classical post processing


![image](https://user-images.githubusercontent.com/41319311/111591097-f44d8d00-8801-11eb-93a8-3804b6b6de88.png)

pauli_class_package:
1) Handles multiplication of pauli strings

Objects: Paulistring
Functions: pauli_combine, pauli_helper

hamiltonian_class_package:
1) Contains all the information of the Hamiltonian

REQUIRES PAULI CLASS

Objects: Hamiltonian
Functions: transverse_ising_model_1d

ansatz_class_package:
1) Generates Ansatzes

Objects: Ansatz, moment
Functions: initial_ansatz, gen_next_ansatz
