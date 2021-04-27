import numpy as np 
import ansatz_class_package as acp 
import pauli_class_package as pcp 
import hamiltonian_class_package as hcp 
import matrix_class_package as mcp 
import post_processing as pp
import plotting_package as plotp
import warnings
import pandas as pd
warnings.filterwarnings("ignore", category=DeprecationWarning) 

#Parameters
uptowhatK = 1
num_qubits = 2
endtime = 8
num_steps = 1001

#create initial state
initial_state = acp.Initialstate(num_qubits, "efficient_SU2", 123, 5)
#define Hamiltonian
# hamiltonian = hcp.transverse_ising_model_1d(num_qubits)
hamiltonian = hcp.heisenberg_xyz_model(num_qubits)

#We can define arbitrary Hamiltonians
#EXAMPLE: If we have a 4 qubit system and I want to implement the hamiltonian H = 0.6*(XXII) + 0.4*(XZIY), the hamiltonian will be generated with this:   
#hcp.generate_arbitrary_hamiltonian(4,[0.6,0.4],["1100","1302"])

import Qiskit_helperfunctions as qhf #IBMQ account is loaded here in this import
hub, group, project = "ibm-q-nus", "default", "reservations"
quantum_com = "ibmq_rome" 

#Other parameters for running on the quantum computer
sim = "noiseless_qasm"
num_shots = 30000

quantum_computer_choice_results = qhf.choose_quantum_computer(hub, group, project, quantum_com)
# mitigate_meas_error = True
# meas_filter = qhf.measurement_error_mitigator(num_qubits, sim, quantum_computer_choice_results, shots = num_shots)

mitigate_meas_error = False
meas_filter = qhf.measurement_error_mitigator(num_qubits, sim, quantum_computer_choice_results, shots = num_shots)

# mitigate_meas_error = False
# meas_filter = None

expectation_calculator = qhf.make_expectation_calculator(initial_state, sim, quantum_computer_choice_results, meas_error_mitigate = mitigate_meas_error, meas_filter = meas_filter)

#Run QAS

#Some extra parameters for QAS
p_invcond = 10**(-6)
optimizer = 'zvode'

#create Initial Ansatz for K = 0
ansatz = acp.initial_ansatz(num_qubits)
#finalresults
finalresults = []
finalresults.append(ansatz)

for k in range(1,uptowhatK+1):
    print(k)

    #Generate Ansatz for this round
    ansatz = acp.gen_next_ansatz(ansatz, hamiltonian, num_qubits)

    #Set initial alphas for Ansatz
    acp.set_initial_alphas(num_qubits,ansatz,'start_with_initial_state')

    E_mat_uneval = mcp.unevaluatedmatrix(num_qubits, ansatz, hamiltonian, "E")
    D_mat_uneval = mcp.unevaluatedmatrix(num_qubits, ansatz, hamiltonian, "D")

    #Here, we can specify how to evaluate the matrices. We can either evaluate the matrices by matrix multiplication,
    # or by using the quantum computer specified above
    
    # E_mat_evaluated =  E_mat_uneval.evaluate_matrix_by_matrix_multiplicaton(initial_state)
    # D_mat_evaluated = D_mat_uneval.evaluate_matrix_by_matrix_multiplicaton(initial_state)

    E_mat_evaluated =  E_mat_uneval.evaluate_matrix_with_qiskit_circuits(expectation_calculator)

    # display(pd.DataFrame(E_mat_uneval.evaluate_matrix_by_matrix_multiplicaton(initial_state)))

    # display(pd.DataFrame(E_mat_evaluated))

    # display(E_mat_uneval.dict_of_uneval_matrix_elems)
    D_mat_evaluated = D_mat_uneval.evaluate_matrix_with_qiskit_circuits(expectation_calculator)

    #Get starting alphas
    startingstrings,startingalphas = ansatz.get_alphas()

    #initialize QAS instance
    QAS_instance = pp.QAS(num_qubits, D_mat_evaluated, E_mat_evaluated, startingalphas)
    QAS_instance.numberstep(num_steps)
    QAS_instance.define_endtime(endtime)
    QAS_instance.define_optimizer(optimizer)
    QAS_instance.define_p_invcond(p_invcond)

    #Run QAS instance
    QAS_instance.evaluate()

    #Get results
    result = QAS_instance.get_results()

    #Update ansatz with the new alphas
    ansatz.update_alphas(result)

    #Update final results with this
    finalresults.append(ansatz)
finalresultsQAS = finalresults


#Run Classical Calculations

#Initialize classicalSimulator
cS_instance = pp.classicalSimulator(num_qubits,initial_state,hamiltonian)
cS_instance.define_endtime(endtime)
cS_instance.numberstep(num_steps)

#Run classicalSimulator
cS_instance.evaluate()

#Observable we want to plot
times = QAS_instance.get_times()
observable = hcp.generate_arbitary_observable(num_qubits, [1], ["30"]) 


#What Ks we want to plot
whatK = [1]

#Plotting results for TTQS
# plotp.QS_plotter_forobservable(num_qubits,finalresultsTTQS,times,whatK,'TTQS',observable,initial_state)

#Plotting results for QAS
plotp.QS_plotter_forobservable(num_qubits,finalresultsQAS,times,whatK,'QAS',observable,initial_state)

#Plotting results for CQFF
# plotp.QS_plotter_forobservable(num_qubits,finalresultsCQFF,times,whatK,'CQFF',observable,initial_state)

#Plotting classical result
observablematrix = observable.to_matrixform()
classicalresult = cS_instance.get_expectations_observables(observablematrix)
plotp.CS_plotter_forobservable(times,classicalresult)

#Show plot
plotp.show_plot()