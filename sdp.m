D = 0.5*(D + D');
E = 0.5*(E + E');

cvx_begin sdp
    variable beta(11,11) hermitian
    minimize(trace(beta*D))
    trace(beta*E)==1;
    rho>=0;
cvx_end
eigenvalues = eig(beta);
disp(eigenvalues)










