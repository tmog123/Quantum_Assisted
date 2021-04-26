import numpy as np 
import ansatz_class_package as acp 
import pauli_class_package as pcp 
import hamiltonian_class_package as hcp 
import matrix_class_package as mcp 
import post_processing as pp
import plotting_package as plotp
import warnings
import Qiskit_helperfunctions_kh as qhf_kh
warnings.filterwarnings("ignore", category=DeprecationWarning) 

#Parameters
uptowhatK = 2
num_qubits = 2
endtime = 8
num_steps = 1001
optimizer = 'eigh'
inv_cond = 10**(-6)
numberoflayers = 3
randomseedforinitialstate = 123

#create initial state
initial_state = acp.Initialstate(num_qubits, "efficient_SU2", randomseedforinitialstate, numberoflayers)

#define Hamiltonian
# hamiltonian = hcp.transverse_ising_model_1d(num_qubits)
hamiltonian = hcp.heisenberg_xyz_model(num_qubits)

#create Initial Ansatz for K = 0
ansatz = acp.initial_ansatz(num_qubits)

#finalresults
finalresults = []
finalresults.append(ansatz)

#First, we generate the ansatz with the highest possible K value, and store all the ansatz that we previously generated.
ansatz_dict = dict()
ansatz_dict[0] = ansatz
for k in range(1,uptowhatK+1):
    print('Currently at K = ' + str(k))

    #Generate Ansatz for this round
    ansatz = acp.gen_next_ansatz(ansatz, hamiltonian, num_qubits) #By default, there is no processing when generating next Ansatz
    ansatz_dict[k] = ansatz

# Next, we calculate the D and E matrix elements for the highest ansatz with
# the highest K we want. The key insight here is that once we do the
# calculation for the Ansatz with the highest K value, we get all the lower K
# value ansatz for free. This is helpful to reduce computations on the quantum
# computer. I.e, if you know you are going to start with K = 3, then why bother
# doing calculations for K = 1 and K = 2?

#Set initial alphas for Ansatz
#Only 'start_with_initial_state' has been implemented thus far. 
#This basically sets the state we want to evolve as the random, initial state we are using to generate the E and D matrices, for convenience
acp.set_initial_alphas(num_qubits,ansatz,'start_with_initial_state')

E_mat_uneval = mcp.unevaluatedmatrix(num_qubits, ansatz, hamiltonian, "E")
D_mat_uneval = mcp.unevaluatedmatrix(num_qubits, ansatz, hamiltonian, "D")

E_mat_uneval_pstring_strforms = E_mat_uneval.return_set_of_pstrings_to_evaluate()
D_mat_uneval_pstring_strforms = D_mat_uneval.return_set_of_pstrings_to_evaluate()
all_uneval_pstring_strforms = E_mat_uneval_pstring_strforms.union(D_mat_uneval_pstring_strforms)

#Here, we evaluate all the uneval_pstrings classically
# initial_statevector = initial_state.get_statevector()
# all_evaluated_pstring_strform_results = mcp.evaluate_pstrings_strings_classicaly(all_uneval_pstring_strforms, initial_statevector)

#Here, we do so in a quantum manner.
num_qubits = num_qubits
num_shots = 8192
quantum_computer = "ibmq_rome"
# sim = "noisy_qasm"
# sim = "noiseless_qasm"
sim = "real"

quantum_computer_choice_results = qhf_kh.choose_quantum_computer(hub = "ibm-q-nus", group = "default", project = "reservations", quantum_com = quantum_computer)
initial_state_object = initial_state

meas_filter = qhf_kh.measurement_error_mitigator(num_qubits, sim, quantum_computer_choice_results, shots = num_shots)
expectation_calculator = qhf_kh.make_expectation_calculator(initial_state_object, sim, quantum_computer_choice_results, meas_error_mitigate=True, meas_filter=meas_filter)

# expectation_calculator = qhf_kh.make_expectation_calculator(initial_state_object, sim, quantum_computer_choice_results, meas_error_mitigate=False)

all_evaluated_pstring_strform_results = dict() 
for pstring_strform in all_uneval_pstring_strforms:
    print("pstring_strform is", pstring_strform)
    pauli_string_object = pcp.paulistring(num_qubits, pstring_strform, 1)
    all_evaluated_pstring_strform_results[pstring_strform] = expectation_calculator(pauli_string_object)


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

    E_mat_evaluated = E_mat_uneval.substitute_evaluated_pstring_results(all_evaluated_pstring_strform_results)

    D_mat_evaluated = D_mat_uneval.substitute_evaluated_pstring_results(all_evaluated_pstring_strform_results)

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
whatK = [1]

#Plotting results
plotp.QS_plotter_forobservable(num_qubits,finalresults,times,whatK,'TTQS',observable,initial_state)

#Plotting classical result
observablematrix = observable.to_matrixform()
classicalresult = cS_instance.get_expectations_observables(observablematrix)
plotp.CS_plotter_forobservable(times,classicalresult)


#Show plot
plotp.show_plot()