D = 0.5*(D + D');
E = 0.5*(E + E');

cvx_begin sdp
    variable rho(11,11) hermitian
    minimize(trace(rho*D))
    trace(rho*E)==1;
    rho>=0;
cvx_end
eigenvalues = eig(rho);
disp(eigenvalues)







