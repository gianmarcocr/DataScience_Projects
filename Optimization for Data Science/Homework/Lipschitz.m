data = readtable('matrix2.csv');
data = table2array(data);
m = length(data);
n = length(data(1, 2:end));
lambda = 0.4;

Q=data(:,2:end); %matrix of regressors
y=data(:,1); %response variable
w = ones(n,1)*0.5;

P_ii = zeros(m,1);

for i=1:m
    x=Q(i,:);x=x';
    P_ii(i) = (1/(1+exp(-y(i)*w'*x))) * (1 - (1/(1+exp(-y(i)*w'*x))));
end
P = diag(P_ii);
beta = max(eig(P));
delta = max(eig(Q' * Q));

L = beta*delta + lambda