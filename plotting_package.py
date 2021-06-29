import matplotlib.pyplot as plt
import numpy as np
import matrix_class_package as mcp
import pauli_class_package as pcp
from scipy.linalg import expm 

def QS_plotter_foralpha(ansatz,times):
    for mom in ansatz.get_moments():
        plt.plot(times, np.abs(mom.alphas[:-1]),label=str(mom.paulistring.string))


def getdata_forbetamatrix_observable(num_qubits,ansatzlist,whatKs,observable,initial_state,betamatrixlist,evalmethod='matrix_multiplication',expectation_calculator=None):
    if evalmethod == "qiskit_circuits" and expectation_calculator == None:
        raise(RuntimeError("You need to pass in the expectation_calculator as an argument (see the Qiskit_helperfunctions package)."))
    allresultlist = []
    #betacounter = 0
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
            result = 0
            for a in range(len(betamatrixlist[i-1])):
                for b in range(len(betamatrixlist[i-1])):
                    result = result + betamatrixlist[i-1][a][b]*Omat[b][a]
            allresultlist.append(result)
            #betacounter = betacounter+1
    return allresultlist
    

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
def get_data_forobservable(num_qubits,ansatzlist,times,whatKs,qstype,observable,initial_state, evalmethod='matrix_multiplication', expectation_calculator = None):
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
    finaldict = {}
    finaldict['times'] = list(times.real)
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
            finaldict[str(i)] = list(observable_vals)
            #plt.plot(times, observable_vals,label=lab)
    return finaldict

def CS_plotter_forobservable(times,classicalresult):
    plt.plot(times,classicalresult,label='Classical')

def show_plot():
    plt.legend()
    plt.show()

def print_plot(location):
    plt.legend()
    plt.savefig(location)
    plt.close()

def QS_plotter_for_fidelity(num_qubits, ansatzlist, times,
     whatKs, qstype, hamiltonian, initial_state):
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
                
                hamiltonian_matrix = hamiltonian.to_matrixform()
                theoretical_state = expm(-1j * hamiltonian_matrix * time) @ initial_statevector
                # theoretical_state = final_results_from_classical_simulator[time_idx]
                fidelity = np.abs(np.vdot(theoretical_state, state))
                # if time == 60:
                #     print("actual_state_is", state)
                #     print("theoretical_state_is", theoretical_state)
                fidelity_vals.append(fidelity)
            lab = name + "Fidelity K=" + str(i)
            plt.plot(times, fidelity_vals, label = lab)

def get_data_for_fidelity(num_qubits, ansatzlist, times,whatKs, qstype, hamiltonian, initial_state):
    finaldict = {}
    finaldict['times'] = list(times.real)
    initial_statevector = initial_state.get_statevector()
    if qstype == 'TTQS':
        name = 'TTQS'
    if qstype == 'QAS':
        name = 'QAS'
    if qstype == 'CQFF':
        name = 'CQFF'
    for i in range(len(ansatzlist)):
        if i in whatKs:
            #print('Plotting fidelity for K = ' + str(i))
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
                
                hamiltonian_matrix = hamiltonian.to_matrixform()
                theoretical_state = expm(-1j * hamiltonian_matrix * time) @ initial_statevector
                # theoretical_state = final_results_from_classical_simulator[time_idx]
                fidelity = np.abs(np.vdot(theoretical_state, state))
                # if time == 60:
                #     print("actual_state_is", state)
                #     print("theoretical_state_is", theoretical_state)
                fidelity_vals.append(fidelity)
            lab = name + " K=" + str(i)
            #plt.plot(times, fidelity_vals, label = lab)
            finaldict[str(i)] = list(fidelity_vals)
    return finaldict


def get_fidelity_results(num_qubits, ansatzlist, times,
    whatKs, hamiltonian, initial_state):
    """
    Returns a dictionary of dictionaries. The keys are the values of K that we
    are considering, the values are the dictionaries for that K value. For a
    particular K value, the corresponding dictionary is such that the keys are
    the time values, and the values are the fidelities at that particular
    time
    """
    initial_statevector = initial_state.get_statevector()
    # if qstype == 'TTQS':
    #     name = 'TTQS'
    # if qstype == 'QAS':
    #     name = 'QAS'
    # if qstype == 'CQFF':
    #     name = 'CQFF'
    collated_fidelity_vals = dict()
    for i in range(len(ansatzlist)):
        if i in whatKs:
            print('Calculating and saving fidelity for K = ' + str(i))
            ansatz = ansatzlist[i]
            result_pauli_string, result_alphas = ansatz.get_alphas()
            result_alphas = list(zip(*result_alphas)) #transpose it so that each entry is a time value 
            fidelity_vals = dict()
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
                
                hamiltonian_matrix = hamiltonian.to_matrixform()
                theoretical_state = expm(-1j * hamiltonian_matrix * time) @ initial_statevector
                # theoretical_state = final_results_from_classical_simulator[time_idx]
                fidelity = np.abs(np.vdot(theoretical_state, state))
                # if time == 60:
                #     print("actual_state_is", state)
                #     print("theoretical_state_is", theoretical_state)
                # fidelity_vals.append(fidelity)
                fidelity_vals[time] = fidelity
            collated_fidelity_vals[i] = fidelity_vals
    return collated_fidelity_vals