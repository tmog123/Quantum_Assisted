import numpy as np 
import ansatz_class_package as acp 
import pauli_class_package as pcp 
import hamiltonian_class_package as hcp 
import matrix_class_package as mcp 
import post_processing as pp
import plotting_package as plotp
import warnings
import json
warnings.filterwarnings("ignore", category=DeprecationWarning) 



#Parameters
uptowhatK = 3
num_qubits = 2
endtime = 5
num_steps = 1001
optimizer ='eigh' #'eigh'
inv_cond = 10**(-3)
numberoflayers = 2
randomseedforinitialstate = 183#183 #873

#create initial state
initial_state = acp.Initialstate(num_qubits, "TFI_hardware_inspired", randomseedforinitialstate, numberoflayers)

#Qiskit stuff
import Qiskit_helperfunctions as qhf #IBMQ account is loaded here in this import
#IBMQ.load_account() 
hub, group, project = "ibm-q-nus", "default", "reservations"
quantum_com = "ibmq_bogota" 

#Other parameters for running on the quantum computer
sim = "noiseless_qasm"# #"noisy_qasm" #"noiseless_qasm"
num_shots = 10000

quantum_computer_choice_results = qhf.choose_quantum_computer(hub, group, project, quantum_com)
# mitigate_meas_error = True
# meas_filter = qhf.measurement_error_mitigator(num_qubits, sim, quantum_computer_choice_results, shots = num_shots)
mitigate_meas_error = False
meas_filter = qhf.measurement_error_mitigator(num_qubits, sim, quantum_computer_choice_results, shots = num_shots)

expectation_calculator = qhf.expectation_calculator(initial_state, sim, quantum_computer_choice_results, meas_error_mitigate = mitigate_meas_error, meas_filter = meas_filter)



#define Hamiltonian
hamiltonian = hcp.transverse_ising_model_1d(num_qubits)

#create Initial Ansatz for K = 0
ansatz = acp.initial_ansatz(num_qubits)

#finalresults
finalresults = []
finalresults.append(ansatz)

#Run TTQS
for k in range(1,uptowhatK+1):
    print('Currently at K = ' + str(k))

    #Generate Ansatz for this round
    ansatz = acp.gen_next_ansatz(ansatz, hamiltonian, num_qubits) #By default, there is no processing when generating next Ansatz

    #Set initial alphas for Ansatz
    #Only 'start_with_initial_state' has been implemented thus far. 
    #This basically sets the state we want to evolve as the random, initial state we are using to generate the E and D matrices, for convenience
    acp.set_initial_alphas(num_qubits,ansatz,'start_with_initial_state')

    E_mat_uneval = mcp.unevaluatedmatrix(num_qubits, ansatz, hamiltonian, "E")
    D_mat_uneval = mcp.unevaluatedmatrix(num_qubits, ansatz, hamiltonian, "D")

    #Here is where we should be able to specify how to evaluate the matrices. However only the exact method (classical matrix multiplication) has been implemented so far
    E_mat_evaluated =  E_mat_uneval.evaluate_matrix_with_qiskit_circuits(expectation_calculator)
    #print(E_mat_evaluated)
    D_mat_evaluated = D_mat_uneval.evaluate_matrix_with_qiskit_circuits(expectation_calculator)
    #print(D_mat_evaluated)

    #Get starting alphas
    startingstrings,startingalphas = ansatz.get_alphas()

    #Initialize TTQS instance
    TTQS_instance = pp.TTQS(num_qubits,D_mat_evaluated,E_mat_evaluated,startingalphas)
    TTQS_instance.numberstep(num_steps)
    TTQS_instance.define_endtime(endtime)
    TTQS_instance.define_optimizer(optimizer)
    TTQS_instance.define_invcond(inv_cond)

    #Run TTQS instance
    TTQS_instance.evaluate()

    #Get results
    result = TTQS_instance.get_results()

    #Update ansatz with the new alphas
    ansatz.update_alphas(result)

    #Update final results with this
    finalresults.append(ansatz)
#print('Length of Ansatz is ' + str(len(finalresults)))

#Run Classical Calculations

#Initialize classicalSimulator
cS_instance = pp.classicalSimulator(num_qubits,initial_state,hamiltonian)
cS_instance.define_endtime(endtime)
cS_instance.numberstep(num_steps)

#Run classicalSimulator
cS_instance.evaluate()


#Now, finalresults is a list of Ansatzes, each Ansatz basically stores the results for that run
#Example, Final results might look like this [Ansatz_0,Ansatz_1,Ansatz_2,Ansatz_3]
#Where Ansatz_1 is an Ansatz class, which was used for the K=1 run, and contains the final results for that run

#Observable we want to plot
times = TTQS_instance.get_times()
observable = hcp.generate_arbitary_observable(num_qubits, [1], ["30"]) 

#What Ks we want to plot
whatK = [1,2,3]

#Plotting results
plotp.QS_plotter_forobservable(num_qubits,finalresults,times,whatK,'TTQS',observable,initial_state,evalmethod = "qiskit_circuits", expectation_calculator = expectation_calculator)

#get data for printing
#ttqsdata = plotp.get_data_forobservable(num_qubits,finalresults,times,whatK,'TTQS',observable,initial_state,evalmethod = "qiskit_circuits", expectation_calculator = expectation_calculator)
ttqsdata = plotp.get_data_for_fidelity(num_qubits,finalresults,times,whatK,'TTQS',hamiltonian,initial_state)


#Plotting classical result
observablematrix = observable.to_matrixform()
classicalresult = cS_instance.get_expectations_observables(observablematrix)
plotp.CS_plotter_forobservable(times,classicalresult)

#get data for printing

ttqsdata['classical'] = list(classicalresult)

#print(ttqsdata)
'''
#Run QAS
p_invcond = 10**(-3)
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

    #Here is where we should be able to specify how to evaluate the matrices. However only the exact method (classical matrix multiplication) has been implemented so far
    E_mat_evaluated =  E_mat_uneval.evaluate_matrix_with_qiskit_circuits(expectation_calculator)
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
    finalresults.append(ansatz)'''


#Plotting results
#plotp.QS_plotter_forobservable(num_qubits,finalresults,times,whatK,'QAS',observable,initial_state)
plotp.QS_plotter_for_fidelity(num_qubits,finalresults,times,whatK,'TTQS',hamiltonian,initial_state)

#Show plot
#plotp.show_plot()
plotp.print_plot("Jonstufftesting/plot.png")
json.dump(ttqsdata, open( "Jonstufftesting/data.dat",'w+'))

# Save data on file
# Prepare a dictionary with the data
dataprint = {}
