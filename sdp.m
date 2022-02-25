

D = 0.5*(D + D');
E = 0.5*(E + E');
gammas = 0.1;
interval = 10^(-4);
numberstate = length(D);


cvx_begin sdp
    variable betarho(numberstate,numberstate) hermitian 
    minimize(0)
    %RHS = zeros(numberstate);
    RHS = -1j*(D*betarho*E - E*betarho*D);
    %disp(RHS)
    for k = 1:length(F(:,1,1))
        %disp(k)
        thisR = squeeze(R(k,:,:));
        thisF = squeeze(F(k,:,:));
        RHS = RHS + gammas*(thisR*betarho*(thisR') - 0.5*thisF*betarho*E - 0.5*E*betarho*thisF);
    end
    %USE THIS LINE FOR HARD CONSTRAINT
    RHS==0;
    %USE THESE LINE FOR INTERVAL CONSTRAINT
    %RHS<interval;
    %RHS>-interval;
    trace(betarho*E)==1;
    betarho>=0;
cvx_end
save('KHstufftesting/savedmatrixfrommatlab.mat','betarho')
eigenvalues = eig(betarho);
disp(eigenvalues)







