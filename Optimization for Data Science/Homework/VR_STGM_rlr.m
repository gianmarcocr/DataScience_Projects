 

function [w,it,fx,ttot,fh,timeVec,gnrit] = VR_STGM_rlr(Q,c,w,lambda,lc,...
    verbosity,nepochs,maxit,eps,fstop,stopcr)



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Implementation of the Stochastic Variance Reduction 
% Gradient Method
%
%INPUTS:
%Q: Q matrix 
%c: c term
%x: starting point
%lc:  constant of the reduced stepsize (numerator) 
%verbosity: printing level
%nepochs: epoch length
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

% gradient evaluation
w0=w;
y=c;
gsvrg=zeros(n,1);
for i=1:m
    x=Q(i,:);x=x';
    gsvrg = gsvrg + ((-y(i)*x*exp(-y(i)*w0'*x))/(1+exp(-y(i)*w0'*x)) );
end
gsvrg=(1/m)*(gsvrg + lambda*w0);

% function evaluation
eta=zeros(m,1);
for i=1:m
    x=Q(i,:);x=x';
    eta(i)=log(1+exp(-y(i)*w'*x));  
end 
fx = (1/m)*(sum(eta) + (lambda/2)*(w'*w));

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
    ind=randi(m);
    %update the gradient for the sample 'ind' 
    g=zeros(n,1);
    x=Q(ind,:);x=x';
    g = g + (-y(ind)*x*exp(-y(ind)*w'*x))/(1+exp(-y(ind)*w'*x)) + lambda*w;
    %compute the 'old' gradient of sample 'ind'
    x=Q(ind,:);x=x';
    gz=(-y(ind)*x*exp(-y(ind)*w0'*x))/(1+exp(-y(ind)*w0'*x)) + lambda*w0;
    gf=g - gz + (1/m)*gsvrg;
    d=-gf;
        
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
        

       
        %alpha selection
        alpha=lc;
        z= w + alpha*d;
        % update gradient at the end of epoch (10000 iterations per epoch)
        if(mod(it,nepochs)==0)
            eta=zeros(m,1);
            for i=1:m
                    x=Q(i,:);x=x';
                    eta(i)=log(1+exp(-y(i)*z'*x));  
            end 
            fz=(1/m)*(sum(eta) + (lambda/2)*(z'*z));
            %update the 'old' gradient
            w0=z;
            gsvrg=zeros(n,1);
            for i=1:m
                x=Q(i,:);x=x';
                gsvrg=gsvrg+(-y(i)*x*exp(-y(i)*z'*x))/(1+exp(-y(i)*z'*x)) + lambda*z ;
            end
            %%%%%
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