import numpy as np 
import ansatz_class_package as acp 
import pauli_class_package as pcp 
import hamiltonian_class_package as hcp 
import matrix_class_package as mcp 
import post_processing as pp
import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning) 

#Parameters
uptowhatK = 3
num_qubits = 3
endtime = 8
num_steps = 1001
optimizer = 'eigh'
eigh_inv_cond = 10**(-12)

#create initial state
initial_state = acp.Initialstate(num_qubits, "efficient_SU2", 123, 5)

#define Hamiltonian
hamiltonian = hcp.heisenberg_xyz_model(num_qubits)

#create Initial Ansatz for K = 0
ansatz = acp.initial_ansatz(num_qubits)

#finalresults
finalresults = []
finalresults.append(ansatz)

#Run CQFF
for k in range(1,uptowhatK+1):
    print(k)

    #Generate Ansatz for this round
    ansatz = acp.gen_next_ansatz(ansatz, hamiltonian, num_qubits)

    #Set initial alphas for Ansatz
    acp.set_initial_alphas(num_qubits,ansatz,'start_with_initial_state')

    E_mat_uneval = mcp.unevaluatedmatrix(num_qubits, ansatz, hamiltonian, "E")
    D_mat_uneval = mcp.unevaluatedmatrix(num_qubits, ansatz, hamiltonian, "D")

    #Here is where we should be able to specify how to evaluate the matrices. However only the exact method (classical matrix multiplication) has been implemented so far
    E_mat_evaluated =  E_mat_uneval.evaluate_matrix_by_matrix_multiplicaton(initial_state)
    D_mat_evaluated = D_mat_uneval.evaluate_matrix_by_matrix_multiplicaton(initial_state)

    #Get starting alphas
    startingstrings,startingalphas = ansatz.get_alphas()

    #initialize CQFF instance
    CQFF_instance = pp.CQFF(num_qubits, D_mat_evaluated, E_mat_evaluated, startingalphas)
    CQFF_instance.numberstep(num_steps)
    CQFF_instance.define_endtime(endtime)
    CQFF_instance.define_optimizer(optimizer)
    CQFF_instance.define_eigh_invcond(eigh_inv_cond)

    #Run CQFF instance
    CQFF_instance.evaluate()

    #Get results
    result = CQFF_instance.get_results()

    #Update ansatz with the new alphas
    ansatz.update_alphas(result)

    #Update final results with this
    finalresults.append(ansatz)


#Now, finalresults is a list of Ansatzes, each Ansatz basically stores the results for that run
#Example, Final results might look like this [Ansatz_0,Ansatz_1,Ansatz_2,Ansatz_3]
#Where Ansatz_1 is an Ansatz class, which was used for the K=1 run, and contains the final results for that run

