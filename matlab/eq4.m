% This function plots the overall value of the objective Function %(Eq 4 in paper). This plot shows how the Function value reaches 
% minimum value after several iterations. 
% U is the membership matrix (262144 x 3)
% C is the cluster centers (3 x 1)
% B is the bias field estimate (512 x 512)
% Written by    1. Awais Ashfaq - KTH 2016
%               2. Jonas Adler - KTH 2016
function [ F ] = eq4( Y,U,c,B,param,Nind)
U=U';
Nr=param.win^2-1;
% Generate Neighbors for all K's
imagesize=size(B);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% s1=0;
% s2=0;
% s3=0;
Y=reshape(Y,[1 numel(Y)]);
if(length(imagesize)==2)
    imagesize(3)=1;
end
B = reshape(B,[1 numel(B)]);
D = (repmat(Y-B,[numel(c) 1]) - (repmat(c',[numel(Y) 1]))').^2;% D_ik
s1 = sum(sum(D.*(U.^2)));
% R_ik
for i=1:numel(c)
    for k=1:numel(B)
        R(i,k)=sum((Y(Nind{k})-B(Nind{k})-repmat(c(i,:),Nr,1)').^2); %R_ir (Eq 7)
    end
end
s2 = sum(sum(R.*(U.^2)));

% s3 =(repmat(1,numel(B),1))' - sum(U); % 3rd term in Eq 7 in paper
% lambda_c=sum(lambda)/-numel(lambda);
% s4=sum(B.*lambda); 4th term in Eq 7 in paper

%Gamma (Eq 14)
% b=0;
% for i=1:numel(c)
%     a = D(i,k) + (param.alpha /size(Nind,1)) .* R(i,k); %Eq 15
%     for j=1:numel(c)
%         b =b+ D(j,k) + (param.alpha /size(Nind,1)) * R(j,k); %Eq 15
%         
%     end
%     if(abs(b)<eps), b=eps; end
%     gamma(k) = param.m/(sum((a/b)^(1/(param.m-1))))^(param.m-1); %Eq 15
% end

F=(s1 + (param.alpha/Nr)*s2);


end

