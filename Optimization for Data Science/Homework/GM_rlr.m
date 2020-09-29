function [w,it,fx,ttot,fh,timeVec,gnrit] = GM_rlr(Q,c,w,lambda,lc,verbosity,maxit,eps,fstop,stopcr)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Implementation of the Gradient Method
%
%INPUTS:
%Q: Q matrix 
%c: c term(=vector y of labels)
%w: starting point
%lambda: shrinkage coeff.
%lc: Lipschitz constant of the gradient
%    (not needed if exact/Armijo ls used)
%verbosity: printing level
%arls: line search (1 Armijo 2 exact 3 fixed)
%maxit: maximum number of iterations
%eps: tolerance
%fstop: target o.f. value
%stopcr: stopping condition
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
gamma=0.0001;
maxniter=maxit;
fh=zeros(1,maxit);
gnrit=zeros(1,maxit);
timeVec=zeros(1,maxit);
flagls=0;

tic;
timeVec(1) = 0;

%Values for the smart computation of the o.f.
m=length(Q);
y=c;
eta=zeros(m,1);
for i=1:m
    x=Q(i,:);x=x';
    eta(i)=log(1+exp(-y(i)*w'*x)) + (lambda/2)*(w'*w);  
end 
fx = (1/m) * (sum(eta));


it=1;


while (flagls==0)
    %vectors updating
    if (it==1)
        timeVec(it) = 0;
    else
        timeVec(it) = toc;
    end
    fh(it)=fx;
    
    % gradient evaluation
    n=length(Q(1,:)); %number of variables(regressors)
    g=zeros(n,1);
    for i=1:m
        x=Q(i,:);x=x';
        g=g+(-y(i)*x*exp(-y(i)*w'*x))/(1+exp(-y(i)*w'*x)) + lambda*w;
    end
    g=(1/m)*g;
    
    d=-g;
        
    gnr = g'*d;
    gnrit(it) = -gnr;
        
        % stopping criteria and test for termination
    if (it>=maxniter)
        break;
    end
        switch stopcr
            case 1
                % continue if not yet reached target value fstop
                if (fx<=fstop)
                    break
                end
            case 2
                % stopping criterion based on the product of the 
                % gradient with the direction
                if (abs(gnr) <= eps)
                    break;
                end
            otherwise
                error('Unknown stopping criterion');
        end % end of the stopping criteria switch
        
        %fixed alpha
        alpha=1/lc;
        z = w + alpha*d;
        for i=1:m
               x=Q(i,:);x=x';
               eta(i)=log(1+exp(-y(i)*z'*x))+(lambda/2)*(z'*z);  
        end 
        fz=(1/m)*(sum(eta));
        

        w=z;
        fx = fz;
        
        
        if (verbosity>0)
            disp(['-----------------** ' num2str(it) ' **------------------']);
            disp(['gnr      = ' num2str(abs(gnr))]);
            disp(['f(x)     = ' num2str(fx)]);
            disp(['alpha     = ' num2str(alpha)]);                    
        end
                     
        it = it+1;
        
        
end

if(it<maxit)
    fh(it+1:maxit)=fh(it);
    gnrit(it+1:maxit)=gnrit(it);
    timeVec(it+1:maxit)=timeVec(it);
end


ttot = toc;


end