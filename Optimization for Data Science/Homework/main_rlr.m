% Optimality tolerance:
eps = 1.0e-3;
% Stopping criterion
%
% 1 : reach of a target value for the obj.func. fk - fstop <= eps
% 2 : \|nabla f(xk)\|^2 <= eps
stopcr = 1;


%verbosity = 0 don't display info, verbosity =1 display info
verb=0;


data = readtable('spam_scaled.csv'); %import data
data=table2array(data);
Q=data(2:end,2:end-1);%matrix of regressors (including intercept)
c=data(2:end,end); %vector of target variable
[m,n] = size(Q);
k= round(0.1*n);

% starting point
x1= ones(n,1);

fstop = 10^-9;
maxit = 300; %num of iterations for GM
maxit2= 5000; %num of iterations for SGM
maxit3= 5000; %num of iterations for SVRGM
lcgm=1; %stepsize for GM
lsg=0.1; %gamma inside stepsize for SGM
lsvrg=0.05; %stepsize for SVRGM
lambda=0.001; %regularization coefficient

        
disp('*****************');
disp('*  GM STANDARD  *');
disp('*****************');


[xgm,itergm,fxgm,tottimegm,fhgm,timeVecgm,gnrgm]=...
GM_rlr(Q,c,x1,lambda,lcgm,verb,maxit,eps,fstop,stopcr);

% Print results:%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf(1,'f(x)  = %10.3e\n',fxgm);
fprintf(1,'Number of non-zero components of x = %d\n',...
    sum((abs(xgm)>=0.000001)));
fprintf(1,'Number of iterations = %d\n',...
    itergm);
fprintf(1,'||gr||^2 = %d\n',...
    gnrgm(maxit));
fprintf(1,'CPU time so far = %10.3e\n', tottimegm);
%%%%%%%%%%t%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('*****************');
disp('*  SGM STANDARD *');
disp('*****************');


[xagm,iteragm,fxsgm,tottimeagm,fhagm,timeVecagm,gnragm]=...
STGM_rlr(Q,c,x1,lambda,lsg,verb,maxit2,eps,fstop,stopcr);

% Print results:%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf(1,'f(x)  = %10.3e\n',fxsgm);
fprintf(1,'Number of iterations = %d\n',...
    iteragm);
fprintf(1,'CPU time so far = %10.3e\n', tottimeagm);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('*****************');
disp('* SVRGM STANDARD *');
disp('*****************');

nepochs=100;
[xagm,iteragm,fxagm,tottimeagm,fhsvrgm,timeVecsvrgm,gnragm]=...
VR_STGM_rlr(Q,c,x1,lambda,lsvrg,verb,nepochs,maxit3,eps,fstop,stopcr);

% Print results:%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf(1,'f(x)  = %10.3e\n',fxagm);
fprintf(1,'Number of iterations = %d\n',...
    iteragm);
fprintf(1,'CPU time so far = %10.3e\n', tottimeagm);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%plot figure
fmin= 0.0;

%plot figure CPU time vs Objective function
figure
semilogy(timeVecgm,fhgm-fmin,'r-')
hold on
semilogy(timeVecagm,fhagm-fmin,'b-')
hold on
semilogy(timeVecsvrgm,fhsvrgm-fmin,'g-')
grid on

title('GD vs SGD vs SVRGM - Objective function')
legend('GM', 'SGM','SVRG')

xlabel('CPU time [s]'); 

ylabel('Objective function');

%plot figure Num of iterations vs Objective function
figure
semilogy(fhgm-fmin,'r-')
hold on
semilogy(fhagm-fmin,'b-')
hold on
semilogy(fhsvrgm-fmin,'g-')
grid on


title('GD vs SGD vs SVRGM - Objective function')
legend('GM', 'SGM','SVRG')

xlabel('Iterations'); 

ylabel('Objective function');




