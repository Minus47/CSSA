function Y = Pre_label(y)
%% y:标签向量；num_l:表示有标签样本的数目；Y:生成的标签矩阵；
%作用是把标签向量y转换成标签矩阵Y
%比如y中的样本类型标签为3，则Y中对应的样本第3列为1、其他列为0

nClass=length(unique(y));
num=length(y);        % num1表示样本的个数
Y_original=zeros(nClass,length(y));    % 原始的标签矩阵全部为零  

for i=1:num
    for j=1:nClass
        if j==y(i)
            Y_original(j,i)=1;   % 为有标签的样本赋标签为1
        end  
    end
end
Y=Y_original';