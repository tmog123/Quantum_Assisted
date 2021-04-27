import matplotlib.pyplot as plt
import numpy as np
import matrix_class_package as mcp
import pauli_class_package as pcp

def QS_plotter_forobservable(num_qubits,ansatzlist,times,whatKs,qstype,observable,initial_state, evalmethod='matrix_multiplication', expectation_calculator = None):
    """
    evalmethod can either be "qiskit_circuits" or "matrix_multiplication"
    """
    if evalmethod == "qiskit_circuits" and expectation_calculator == None:
        raise(RuntimeError("You need to pass in the expectation_calculator as an argument (see the Qiskit_helperfunctions package)."))
    if qstype == 'TTQS':
        name = 'TTQS'
    if qstype == 'QAS':
        name = 'QAS'
    if qstype == 'CQFF':
        name = 'CQFF'
    for i in range(len(ansatzlist)):
        if i in whatKs:
            print('Preparing observable for plotting for K = ' + str(i))
            ansatz = ansatzlist[i]
            O_matrix_uneval = mcp.unevaluatedmatrix(num_qubits, ansatz, observable, "O")
            if evalmethod == 'matrix_multiplication':
                print('Evaluating Observable Matrix classically')
                Omat = O_matrix_uneval.evaluate_matrix_by_matrix_multiplicaton(initial_state)
            elif evalmethod == "qiskit_circuits":
                print("Evaluating Observable Matrix with Qiskit circuits")
                Omat = O_matrix_uneval.evaluate_matrix_with_qiskit_circuits(expectation_calculator)
            result_pauli_string, result_alphas = ansatz.get_alphas() 
            result_alphas = list(zip(*result_alphas))
            observable_vals = []
            for time_idx in range(len(times)):
                time = times[time_idx]
                alpha = result_alphas[time_idx] 
                alpha = np.array(alpha)
                observable_value = alpha.conj().T @ Omat @ alpha
                observable_value = observable_value.real 
                observable_vals.append(observable_value)
            lab = name + ' K=' + str(i)
            plt.plot(times, observable_vals,label=lab)

def CS_plotter_forobservable(times,classicalresult):
    plt.plot(times,classicalresult,label='Classical')

def show_plot():
    plt.legend()
    plt.show()

def print_plot(location):
    plt.legend()
    plt.savefig(location)

def QS_plotter_for_fidelity(num_qubits, ansatzlist, times,
     whatKs, qstype, final_results_from_classical_simulator, initial_state):
    initial_statevector = initial_state.get_statevector()
    if qstype == 'TTQS':
        name = 'TTQS'
    if qstype == 'QAS':
        name = 'QAS'
    if qstype == 'CQFF':
        name = 'CQFF'
    for i in range(len(ansatzlist)):
        if i in whatKs:
            print('Plotting fidelity for K = ' + str(i))
            ansatz = ansatzlist[i]
            result_pauli_string, result_alphas = ansatz.get_alphas()
            result_alphas = list(zip(*result_alphas)) #transpose it so that each entry is a time value 
            fidelity_vals = []
            for time_idx in range(len(times)):
                time = times[time_idx]
                alpha = result_alphas[time_idx]
                alpha = np.array(alpha)
                state = np.zeros(2**num_qubits) + 1j*np.zeros(2**num_qubits)
                for j in range(len(alpha)):
                    # print("I'm here", result_pauli_string[j])
                    result_pauli_string_obj = pcp.paulistring(num_qubits, result_pauli_string[j], 1)
                    # print(result_pauli_string_obj)
                    result_pauli_string_matrix = result_pauli_string_obj.get_matrixform()
                    # print("i am here", len(initial_statevector))
                    # print("i am here two", len(result_pauli_string_matrix))
                    # print("i am here three", alpha[j])
                    state += alpha[j] * result_pauli_string_matrix @initial_statevector
                theoretical_state = final_results_from_classical_simulator[time_idx]
                fidelity = np.sqrt(np.vdot(theoretical_state, state))
                fidelity_vals.append(fidelity)
            lab = name + " K=" + str(i)
            plt.plot(times, fidelity_vals, label = lab)