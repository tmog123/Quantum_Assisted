import matplotlib.pyplot as plt
import numpy as np
import matrix_class_package as mcp

def QS_plotter_forobservable(num_qubits,ansatzlist,times,whatKs,qstype,observable,initial_state,evalmethod='matrix_multiplication'):
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

