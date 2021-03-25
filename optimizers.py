import numpy as np
import scipy


def diag_routine(D_matrix, E_matrix, inv_cond = 10**(-2)):
    #Like toby's diag routine, with some modifications (return all the eigvecs rather than just the first one)
    #inversion cutoff, e.g. eigenvalues smaller than this are set to zero
    #this value can be adjusted depending on shot noise, try to find value that fits your noise level
    # inv_cond=10**-12 #i use this value for statevector simulation (no shotnoise)
    # inv_cond=10**-2 #i use this value for shotnoise with ~10000 shots
    #Here, I'll use eigh directly because now I'm always solving the Hermitian version. I.e, D_matrix must be hermitian

    e_vals,e_vecs=scipy.linalg.eigh(E_matrix)
            
    #get e_matrix eigenvalues inverted, cutoff with inv_cond
    e_vals_inverted=np.array(e_vals)

    for k in range(len(e_vals_inverted)):
        if(e_vals_inverted[k]<inv_cond):
            e_vals_inverted[k]=0
        else:
            e_vals_inverted[k]=1/e_vals_inverted[k]
            
    #get e_matrix eigenvalues conditioned, such that small/negative eigenvalues are set to zero
    e_vals_cond=np.array(e_vals)
    for k in range(len(e_vals_cond)):
        if(e_vals_cond[k]<inv_cond):
            e_vals_cond[k]=0

    #convert generalized eigenvalue problem with a regular eigenvalue problem using paper "EIGENVALUE PROBLEMS IN STRUCTURAL MECHANICS"
    #we want to solve D\alpha=\lambda E\alpha
    #turns out this does not work well if E_matrix has near zero eigenvalues
    #instead, we turn this into regular eigenvalue problem which is more behaved
    #we diagonalize E_matrix=U*F*F*U^\dag with diagonal F
    #Then, define S=U*F, and S^-1=F^-1*U^\dag. Use conditioned eigenvalues F for this such that no negative eigenvalues appear, and for inverse large eigenvalues set to zero
    #solve S^-1*D*S^-1^\dag*a=\lambda a
    #convert \alpha=S^-1^\dag*a. This is the solution to original problem.
    #this procedure ensures that converted eigenvalue problem remains hermitian, and no other funny business happens

    # s_matrix = e_vecs@np.diag(np.sqrt(e_vals_cond))
    s_matrix_inv = np.diag(np.sqrt(e_vals_inverted))@e_vecs.conj().T
    toeigmat = s_matrix_inv @ D_matrix @ s_matrix_inv.conj().T

    #S^-1*D*S^-1^\dag matrix might not be Hermitian if D is not Hermitian. SO use
    #eig instead of eigh
    #We still use the term "qae_energy" because traditionally this is where the generalised eigenvalue problem came from
    qae_energy,qae_vectors=scipy.linalg.eigh(toeigmat)

    ini_alpha_vecs = qae_vectors
    ini_alpha_vecs = s_matrix_inv.conj().T @ ini_alpha_vecs

    #Note that after the above procedure, since some of the eigenvectors are in the null-space of the E_matrix (recall that we mapped those "wrong" eigenvalues to 0),
    #those eigenvectors are not what we want. So, we need to throw them away.
    #To find those eigenvectors corresponding to these wrong eigenvalues, because they are in the nullspace of E, they correspond to alphavec^\dag E alphavec = 0
    #Here, we might as well normalise the alpha_vecs,too because they might be
    #like wrong, cause we kinda threw values away
    correct_eigvals = []
    first_index = 0
    for j in range(len(qae_energy)):
        jth_vector = ini_alpha_vecs[:,j]
        norm = np.sqrt(jth_vector.conj().T @ E_matrix @ jth_vector)
        if np.abs(1-norm) < inv_cond:
            first_index = j 
            break 
    first_vector = ini_alpha_vecs[:,first_index]
    correct_eigvals.append(qae_energy[first_index])
    after_normalisation = first_vector / np.sqrt(first_vector.conj().T @ E_matrix @ first_vector)

    for j in range(first_index + 1, len(qae_energy)):
        jth_vector = ini_alpha_vecs[:,j]
        norm = np.sqrt(jth_vector.conj().T @ E_matrix @ jth_vector)
        if np.abs(1-norm) < inv_cond:
            jth_vector = jth_vector/norm
            correct_eigvals.append(qae_energy[j])
            after_normalisation = np.column_stack((after_normalisation,
            jth_vector))

    return (np.array(correct_eigvals) ,after_normalisation)

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