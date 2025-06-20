%% min ||Ps*Xs-Ys||+||Pt*Xt-Vt||+beta*||Gt-Vt'*Vt||+lambda*||Ps-Pt||+gamma*Tr(P*X*L*X'*P')+alpha*(||Ps'||2,1+||Pt'||2,1)
function [P,Ps,Pt,obj] = CSSA(Xs,Xt,Ys,Gt,L,options,obj,loop)
    alpha=options.alpha;
    beta=options.beta;
    lambda=options.lambda;
    gamma=options.gamma;
    dim=options.dim;

    Xs=Xs';Xt=Xt';
    X=[Xs,Xt];
    Ys= Pre_label(Ys);
    Ys=Ys';
    normX=norm(X,'fro')^2;

%% initialization
    option = [];
    option.ReducedDim = dim;
    [Ps,~] = PCA1(Xs', option);
    Ps=Ps';
    [Pt,~] = PCA1(Xt', option);
    Pt=Pt';
    
%     Vt = 2 * full(sqrt(mean(mean(Gt)) / dim)) * rand(size(Xt,2), dim);
%     [~,Vt]=litekmeans(Xt,dim);
%     Vt=Vt';
    Vt=Pt*Xt;
    
    XS=Xs*Xs';
    XT=Xt*Xt';
    XLX=X*L*X';
    YX=Ys*Xs';
    E=eye(size(X,1),size(X,1));
    
    iter = 1;
    maxIter = 30;
    
%% iteration
    while iter <= maxIter
        
        %update Pt
        Wti=sqrt(sum(Pt'.*Pt',2)+eps);
        d = 0.5./Wti;
        Wt = diag(d);
        clear d;
        
        up = 4*lambda*Ps+4*Vt*Xt'-gamma*Ps*XLX;
        down = 4*XT+4*lambda*E+gamma*XLX+4*alpha*Wt;
        Pt=up/down;
        clear up down;
        
        %update Ps
        Wsi = sqrt(sum(Ps'.*Ps',2)+eps);
        d = 0.5./Wsi;
        Ws = diag(d);
        clear d;
        
        up = 4*YX+4*lambda*Pt-gamma*Pt*XLX;
        down = 4*XS+4*lambda*E+gamma*XLX+4*alpha*Ws;
        Ps=up/down;
        clear up down;
        
        %update Vt
        up=2*Pt*Xt+4*beta*Vt*Gt;
        down=2*Vt+4*beta*Vt*Vt'*Vt;
        Vtgradient=up./down;
        Vt=Vt.*Vtgradient;
        clear up down;
        
        I1=norm(Ps*Xs-Ys,'fro')^2;
        I2=norm(Pt*Xt-Vt,'fro')^2;
        I3=norm(Gt-Vt'*Vt,'fro')^2;
        I4=norm(Ps-Pt,'fro')^2;
        P=(Ps+Pt)/2;
        I5=trace(P*XLX*P');
        I6=trace(Ps*Ws*Ps')+trace(Pt*Wt*Pt');

        obj(loop,iter)=(I1+I2+beta*I3+lambda*I4+gamma*I5+alpha*I6)/normX;
        if iter >3 && abs(obj(loop,iter)-obj(loop,iter-1))<1e-3
            break;
        end
        iter = iter + 1;
    end
end