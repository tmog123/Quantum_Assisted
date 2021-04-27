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

#%%
# ### Set parameters for the evaluation
uptowhatK = 4
num_qubits = 3
endtime = 8
num_steps = 1001

# %%
#create initial state
initial_state = acp.Initialstate(num_qubits, "efficient_SU2", 123, 5)
#define Hamiltonian
hamiltonian = hcp.transverse_ising_model_1d(num_qubits)
# hamiltonian = hcp.heisenberg_xyz_model(num_qubits)

# %%
# ### Define the variables needed for computation on the quantum computer
import Qiskit_helperfunctions as qhf #IBMQ account is loaded here in this import
hub, group, project = "ibm-q-nus", "default", "reservations"
quantum_com = "ibmq_rome" 

#Other parameters for running on the quantum computer. Choose 1 to uncomment.
# sim = "noiseless_qasm"
# num_shots = 30000 #max is 1000000

sim = "noisy_qasm"
num_shots = 8192 #max is 8192

# sim = "real"
# num_shots = 8192 #max is 8192

quantum_computer_choice_results = qhf.choose_quantum_computer(hub, group, project, quantum_com)

mitigate_meas_error = True 
meas_filter = qhf.measurement_error_mitigator(num_qubits, sim, quantum_computer_choice_results, shots = num_shots)

# mitigate_meas_error = False
# meas_filter = None

#expectation calculator here is an object that has a method that takes in a paulistring object P, and returns a <psi|P|psi>.
#This expectation calculator also stores previously calculated expectation values, so one doesn't need to compute the same expectation value twice.
expectation_calculator = qhf.expectation_calculator(initial_state, sim, quantum_computer_choice_results, meas_error_mitigate = mitigate_meas_error, meas_filter = meas_filter)

# %%
#Some extra parameters for CQFF
optimizer = 'eigh'
eigh_inv_cond = 10**(-12)

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

    #Here, we can specify how to evaluate the matrices. We can either evaluate the matrices by matrix multiplication,
    # or by using the quantum computer specified above
    
    E_mat_evaluated =  E_mat_uneval.evaluate_matrix_by_matrix_multiplicaton(initial_state)
    D_mat_evaluated = D_mat_uneval.evaluate_matrix_by_matrix_multiplicaton(initial_state)

    # E_mat_evaluated =  E_mat_uneval.evaluate_matrix_with_qiskit_circuits(expectation_calculator)
    # D_mat_evaluated = D_mat_uneval.evaluate_matrix_with_qiskit_circuits(expectation_calculator)

    #Get starting alphas
    startingstrings,startingalphas = ansatz.get_alphas()

    #initialize CQFF instance
    # CQFF_instance = pp.CQFF(num_qubits, D_mat_evaluated, E_mat_evaluated, startingalphas, method="U_dt")
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
finalresultsCQFF = finalresults

# %%
#Run Classical Calculations

#Initialize classicalSimulator
cS_instance = pp.classicalSimulator(num_qubits,initial_state,hamiltonian)
cS_instance.define_endtime(endtime)
cS_instance.numberstep(num_steps)

#Run classicalSimulator
cS_instance.evaluate()

# %%
#Observable we want to plot
times = CQFF_instance.get_times()
observable = hcp.generate_arbitary_observable(num_qubits, [1], ["3" + (num_qubits - 1) * "0"]) 

# %%
#What Ks we want to plot
whatK = [1,2]

#Here, we can either evaluate the O matrix classically, or with qiskit circuits

#Plotting results for CQFF
plotp.QS_plotter_forobservable(num_qubits,finalresultsCQFF,times,whatK,'CQFF',observable,initial_state, evalmethod = "matrix_multiplication")
# plotp.QS_plotter_forobservable(num_qubits,finalresultsCQFF,times,whatK,'CQFF',observable,initial_state, evalmethod = "qiskit_circuits", expectation_calculator = expectation_calculator)

#Plotting classical result
observablematrix = observable.to_matrixform()
classicalresult = cS_instance.get_expectations_observables(observablematrix)
plotp.CS_plotter_forobservable(times,classicalresult)

#Show plot
plotp.show_plot()