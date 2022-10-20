import numpy as np
import ansatz_class_package as acp 
import pauli_class_package as pcp 
import hamiltonian_class_package as hcp 
import matrix_class_package as mcp 
import post_processing as pp
import scipy as scp
import scipy.io
import qutip 
import plotting_package as plotp
import matplotlib.pyplot as plt

g_vals = [0,0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
num_qubits = 8

def generate_fuji_boy_hamiltonian(num_qubits, g):
    epsilon = 0.5
    if num_qubits == 1:
        hamiltonian = hcp.generate_arbitary_hamiltonian(num_qubits,[epsilon,g],['3','1'])
    else:
        hamiltonian = hcp.transverse_ising_model_1d(num_qubits, -0.5, g)
    return hamiltonian

def generate_fuji_boy_gamma_and_Lterms(num_qubits):
    gammas_to_append = 1
    gammas = []
    L_terms = []
    if num_qubits == 1:
        gammas.append(gammas_to_append)
        L_terms.append(hcp.generate_arbitary_hamiltonian(1, [0.5,0.5j],["1","2"]))
    else:
        for i in range(num_qubits):
            gammas.append(gammas_to_append)
            L_terms.append(hcp.generate_arbitary_hamiltonian(num_qubits,[1],['0'*i+'3'+'0'*(num_qubits-1-i)]))
            gammas.append(gammas_to_append)
            L_terms.append(hcp.generate_arbitary_hamiltonian(num_qubits,[0.5,-0.5j],['0'*i+'1'+'0'*(num_qubits-1-i),'0'*i+'2'+'0'*(num_qubits-1-i)]))
    return (gammas, L_terms)


# def _blob(x, y, area, colour):
#     """
#     Draws a square-shaped blob with the given area (< 1) at
#     the given coordinates.
#     """
#     hs = np.sqrt(area) / 2
#     xcorners = np.array([x - hs, x + hs, x + hs, x - hs])
#     ycorners = np.array([y - hs, y - hs, y + hs, y + hs])
#     plt.fill(xcorners, ycorners, colour, edgecolor=colour)

# def hinton(W, maxweight=None):
#     """
#     Draws a Hinton diagram for visualizing a weight matrix. 
#     Temporarily disables matplotlib interactive mode if it is on, 
#     otherwise this takes forever.
#     """
#     reenable = False
#     if plt.isinteractive():
#         plt.ioff()
#     reenable = True
#     plt.clf()
#     height, width = W.shape
#     if not maxweight:
#         maxweight = 2**np.ceil(np.log(np.max(np.abs(W)))/np.log(2))
        
#     plt.fill(np.array([0, width, width, 0]), 
#              np.array([0, 0, height, height]),
#              'gray')
    
#     plt.axis('off')
#     plt.axis('equal')
#     for x in range(width):
#         for y in range(height):
#             _x = x+1
#             _y = y+1
#             w = W[y, x]
#             if w > 0:
#                 _blob(_x - 0.5,
#                       height - _y + 0.5,
#                       min(1, w/maxweight),
#                       'white')
#             elif w < 0:
#                 _blob(_x - 0.5,
#                       height - _y + 0.5, 
#                       min(1, -w/maxweight), 
#                       'black')
#     if reenable:
#         plt.ion()

for g in g_vals:
    print(g)
    hamiltonian = generate_fuji_boy_hamiltonian(num_qubits, g)
    gammas, L_terms = generate_fuji_boy_gamma_and_Lterms(num_qubits)
    qtp_hamiltonian = qutip.Qobj(hamiltonian.to_matrixform())
    qtp_Lterms = [qutip.Qobj(i.to_matrixform()) for i in L_terms]
    qtp_C_ops = [np.sqrt(gammas[i]) * qtp_Lterms[i] for i in range(len(qtp_Lterms))]
    qtp_rho_ss = qutip.steadystate(qtp_hamiltonian, qtp_C_ops,method="iterative-gmres",maxiter=10000)
    # matrix = scp.sparse.csr_matrix(qtp_rho_ss.data)
    matrix = qtp_rho_ss.full()
    # print(type(matrix))
    matrix_real = np.real(matrix)
    matrix_imag = np.imag(matrix)
    
    fig, ax = plt.subplots()
    img = ax.imshow(matrix_real,cmap='RdYlGn', interpolation='nearest')
    clb = plt.colorbar(img)
    # plt.show()
    clb.ax.tick_params(labelsize=8) 
    clb.ax.set_title('Real g=%s'%(g),fontsize=8)
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.xlabel('h1')
    # plt.xticks(np.linspace(0,matrix_real.shape[0]-1,11,endpoint=True),np.round(np.linspace(0,1.6,11,endpoint=True),3))
    # plt.ylabel('%s'%(r'$\Omega$'))
    # plt.ylabel('h2')
    # plt.yticks(np.linspace(0,matrix_real.shape[0]-1,11,endpoint=True),np.round(np.linspace(-1.6,1.6,11,endpoint=True),3))
    plt.savefig('reverse_engineer_ansatz_results/hintonreal_qubits%s_g%s.png'%(num_qubits,g))
    plt.close()

    fig, ax = plt.subplots()
    img = ax.imshow(matrix_imag,cmap='RdYlGn', interpolation='nearest')
    clb = plt.colorbar(img)
    # plt.show()
    clb.ax.tick_params(labelsize=8) 
    clb.ax.set_title('Imag g=%s'%(g),fontsize=8)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('reverse_engineer_ansatz_results/hintonimag_qubits%s_g%s.png'%(num_qubits,g))
    plt.close()



