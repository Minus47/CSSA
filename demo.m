clear;
clc;
addpath('./libsvm-new');
addpath('./dataset');
warning off;
for testnum=1
    scale=0.7;
    switch testnum
        case 1
            name='be-CVE';
            load('berlin_feature');
            load('berlin_label');
            Xs=double(feature); clear feature
            Ys=double(label);   clear label
            load('CVE_feature');
            load('CVE_label');
        otherwise
            break;
    end
    
    nt=size(feature,1);
    Xtrain=double(feature(1:round(nt*scale),:));    %rand round
    Ytrain=double(label(1:round(nt*scale),:));
    Xtest=double(feature(round(nt*scale)+1:end,:)); clear feature
    Yreal=double(label(round(nt*scale)+1:end,:));   clear label
    
    Xs=normalization(Xs',1);
    Xs=Xs';
    Xtrain=normalization(Xtrain',1);
    Xtrain=Xtrain';
    Xtest=normalization(Xtest',1);
    Xtest=Xtest';

    X=[Xs;Xtrain;Xtest];
    
    [COEFF,SCORE, latent] = pca(X);
    SelectNum = cumsum(latent)./sum(latent);
    index = find(SelectNum >= 0.98);
    pca_dim = index(1);
    X=SCORE(:,1:pca_dim);
    
    Xs = X(1:size(Xs,1),:);
    Xtrain = X(size(Xs,1)+1:size(Xs,1)+size(Xtrain,1),:);
    Xtest = X(size(Xs,1)+size(Xtrain,1)+1:end,:);
    
    Y=[Ys;Ytrain];
    n_class=size(unique(Y),1);
    
    options=[];
    options.NeighborMode='KNN';
    options.WeightMode='HeatKernel';
    options.k=5;
    options.dim=n_class;
    
    X=[Xs;Xtrain];
    S = full(constructW(X,options));
    D=diag(sum(S));
    L=D-S;

    delta=1;
    Gt=[];
    
    Nt=size(Xtrain,1);
    for i=1:Nt
        for j=1:Nt
            Gt(i,j)=exp(-norm(Xtrain(i,:)-Xtrain(j,:),2)^2)/(2*delta);
        end
    end
    
    loop=0;
    acc=0;
    acc_max=0;
    acc_res=[];
    cls_res=[];
    obj=[]; 
    
    %% experiments
    p=[0.001,0.01,0.1,1,10,100,1000];

    for alpha=p
        for beta=p
            for lambda=p
                for gamma=p
                loop=loop+1;
                
                options.alpha=alpha;
                options.beta=beta;
                options.lambda=lambda;
                options.gamma=gamma;

                [P,Ps,Pt,obj] = CSSA(Xs,Xtrain,Ys,Gt,L,options,obj,loop);
                
                Zt=P*Xtest';
                Zt = Zt*diag(sparse(1./sqrt(sum(Zt.^2))));
                [~,cls] = max(Zt',[],2);
                acc=mean(Yreal == cls)*100;
                acc_res(loop,2) = mean(Yreal == cls)*100;
                
                acc_res(loop,1)=loop;
                acc_res(loop,3)=alpha;
                acc_res(loop,4)=beta;
                acc_res(loop,5)=lambda;
                acc_res(loop,6)=gamma;
                
                cls_res(:,loop)=cls;
                
                if acc>acc_max
                    acc_max=acc;
                end

                msg=[name,'   ','loop: ',num2str(loop),'   acc: ',num2str(acc_res(loop,2)),'   acc: ',num2str(acc_max)];
                disp(msg);
                
                end
            end
        end
    end
end
