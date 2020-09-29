 

function [w,it,fx,ttot,fh,timeVec,gnrit] = STGM_rlr(Q,c,w,lambda,lc,...
    verbosity,maxit,eps,fstop,stopcr)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Implementation of the Stochastic Gradient Method
%
%INPUTS:
%Q: Q matrix 
%c: c term
%x: starting point
%lc:  constant of the reduced stepsize (numerator) 
%verbosity: printing level
%maxit: maximum number of iterations
%eps: tolerance
%fstop: target o.f. value
%stopcr: stopping condition
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        

maxniter=maxit;
fh=zeros(1,maxit);
gnrit=zeros(1,maxit);
timeVec=zeros(1,maxit);
flagls=0;


tic;
timeVec(1) = 0;

[m,n] = size(Q);

%Values for the smart computation of the o.f.
y=c;
eta=zeros(m,1);
for i=1:m
    x=Q(i,:);x=x';
    eta(i)=log(1+exp(-y(i)*w'*x)) +(lambda/2)*(w'*w);  
end 
fx =(1/m)* (sum(eta)) ;

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
    ind = randi(m);
    g = zeros(n,1);
    x = Q(ind,:);x=x';
    g = g + (-y(ind)*x*exp(-y(ind)*w'*x))/(1+exp(-y(ind)*w'*x)) + lambda*w;

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
        

       
        %reduced alpha
        alpha =sqrt(lc/(it+1));
        z= w + alpha*d;
        if(mod(it,50)==0)
            for i=1:m
                    x=Q(i,:);x=x';
                    eta(i)=log(1+exp(-y(i)*z'*x)) +(lambda/2)*(z'*z);  
            end 
            fz=(1/m)*(sum(eta));
        else
            
            fz=fx;
        end
        
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