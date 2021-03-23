import numpy as np
import scipy


#OPTIMIZERS MUST RETURN THE UPDATED ALPHAS
def eigh_method_for_TTQS(E_matrix,W_matrix,alphas,inv_cond):
    e_vals,e_vecs=scipy.linalg.eigh(E_matrix)
    e_vals_inverted=np.array(e_vals)
    e_vals_cond=np.array(e_vals)
    for k in range(len(e_vals_cond)):
        if(e_vals_cond[k]<inv_cond):
            e_vals_cond[k]=0
    W_matrix = -W_matrix
    #convert generalized eigenvalue problem with a regular eigenvalue problem using paper "EIGENVALUE PROBLEMS IN STRUCTURAL MECHANICS"
    #we want to solve W\alpha=\lambda E\alpha
    #turns out this does not work well if E_matrix has near zero eigenvalues
    #instead, we turn this into regular eigenvalue problem which is more behaved
    #we diagonalize E_matrix=U*F*F*U^\dag with diagonal F
    #Then, define S=U*F, and S^-1=F^-1*U^\dag. Use conditioned eigenvalues F for this such that no negative eigenvalues appear, and for inverse large eigenvalues set to zero
    #solve S^-1*W*S^-1^\dag*a=\lambda a
    #convert \alpha=S^-1^\dag*a. This is the solution to original problem.
    #this procedure ensures that converted eigenvalue problem remains hermitian, and no other funny business happens
    s_matrix=np.dot(e_vecs,np.diag(np.sqrt(e_vals_cond)))
    s_matrix_inv=np.dot(np.diag(np.sqrt(e_vals_inverted)),np.transpose(np.conjugate(e_vecs)))
    toeigmat=np.dot(s_matrix_inv,np.dot(W_matrix,np.transpose(np.conjugate(s_matrix_inv))))
    energy,vectors=scipy.linalg.eigh(toeigmat)
    #energy,vectors=scipy.linalg.eig(toeigmat)
    #print(energy)
    #smallestindex = 0
    #minimumev = np.real(energy[0])
    #for i in range(len(energy)):
    #    if np.real(energy[i])<minimumev:
    #        smallestindex = i
    #        minimumev = np.real(energy[i])
    #print(smallestindex)
    ini_alpha_vec=vectors[:,0]
    ini_alpha_vec=np.dot(np.transpose(np.conjugate(s_matrix_inv)),ini_alpha_vec)
    norm_ini_alpha=np.sqrt(np.abs(np.dot(np.transpose(np.conjugate(ini_alpha_vec)),np.dot(E_matrix,ini_alpha_vec))))
    newalpha=ini_alpha_vec/norm_ini_alpha
    return newalpha
