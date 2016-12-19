% This function implements the methods section of our paper and Ahmed's
% referenced in the main file (runMe).
% Written by    1. Awais Ashfaq - KTH 2016
%               2. Jonas Adler - KTH 2016

function [B,U,bias_mask,U_mask,c,F]=mfcm2(Y,c,Options,var,debug)
Y_im=Y; %Make copy
h=1;
%% Process inputs
defaultoptions=struct('p',2,'alpha',1,'epsilon',1e-5,'sigma',25,'maxit',10);
if(~exist('Options')),
    Options=defaultoptions;
end

%% Step 1, intialization
if(~isa(Y,'double'))
    error('Input image must be double');
end
% Constant in FCM objective function , must be larger than 1
p = Options.m;
% Effect of neighbors
alpha = Options.alpha;
% Store input image dimensions
imagesize=size(Y);
if(length(imagesize)==2)
    imagesize(3)=1;
end
% Convert image to long array
Y=reshape(Y,[numel(Y) size(Y,3)]);
% Stop if difference between current and previous class prototypes is
% smaller than epsilon
epsilon = Options.epsilon;
% Bias field Gaussian smoothing sigma
sigma= Options.sigma;
% Maximun number of iterations
maxit = Options.maxit;
% Previous class prototypes (means)
c_old=zeros(size(c));
% Number of classes
C=length(c);
% Bias field estimate initialization to a small random value
B = randn(size(Y)).*0.001;
% B=ones(size(Y)).*1e-4; Number of pixels
N=size(Y,1);
% Partition matrix
U = zeros([C N]);
Up = zeros([C N]);
% Distance to clusters
D = zeros(1,C);
% Neighbour coordinates of a pixel
Ne=[-1 -1; -1  0; -1  1; 0 -1;  0  1; 1 -1;  1  0;  1  1];
Nr=8;
% Neighborhood window : win*win is the window
win =Options.win;
if win==3;
    for k=1:numel(B)
        [x,y] = ind2sub(imagesize(1:2),k);
        x=min(max(x,2),imagesize(1)-1); y=min(max(y,2),imagesize(2)-1);
        Nx=repmat(x,[Nr 1])+Ne(:,1); Ny=repmat(y,[Nr 1])+Ne(:,2);
        Nind{k} = sub2ind(imagesize(1:2),Nx,Ny); %These are neighbor indixes 8 x 262144 (for a window size of 3 x 3). It speeds up the processing.
    end
else
    % Get neighbour pixel indices if window is bigger than 3 x 3
    for k=1:numel(B)
        [a b]=ind2sub(size(Y_im),k);
        Data_coord=horzcat(a,b);
        Nind{k}=a_get2dpatch(Y_im,Data_coord,win);  %These are neighbor indixes. (win^2 - 1 x 262144)
    end
    Nr=win^2-1;
end
% Neighbour class influence
Gamma = zeros(1,C);
itt=1; %Initialize iteration count

%%%%% Begin Iterations    %%%%%%%%%%

while((sum(sum((c-c_old).^2))>=epsilon)&&(itt<=maxit)), % Stopping criteria (Eq 19 in Paper)
    % while(itt<=maxit),
    disp(['iteration ' num2str(itt)]);
    % Cluster update storage
    num_c=zeros(C,imagesize(3)); den_c=zeros(C,imagesize(3));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Loop through all pixels
    for k=1:N
        % Calculate Dik and Rik as given in Eq.7 in paper
        for i=1:C
            %             Gamma(i)=sum(sum((Y(Nind,:)-B(Nind,:)-repmat(c(i,:),Nr,1)).^2));
            %             D(i) = sum( (Y(k,:)-B(k,:)-c(i,:)).^2 );
            Gamma(i)=sum((Y(Nind{k},:)-B(Nind{k},:)-repmat(c(i,:),Nr,1)).^2);
            D(i) = (Y(k)-B(k)-c(i)).^2 ;
            
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Used in Eq 16 in paper: independent of c
        s = (Y(k)-B(k)) + (alpha / Nr)*sum(Y(Nind{k})-B(Nind{k}));
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % For all Clusters Update Partition Matrix: Eq 15 in paper
        for i=1:C
            dent=0;
            a = D(i) + (alpha /Nr) * Gamma(i); %Eq 15
            for j=1:C
                b = D(j) + (alpha /Nr) * Gamma(j); %Eq 15
                if(abs(b)<eps)
                    b=eps;
                end
                dent = dent + (a/b)^(1/(p-1)); %Eq 15
            end
            if(abs(dent)<eps)
                dent=eps;
            end
            U(i,k) = 1 / dent;
            Up(i,k) = U(i,k).^p; %Eq 15
            
            % To be used in Eq 16
            num_c(i,:)=num_c(i,:)+Up(i,k)*s; %Eq 16
            %             den_c(i)=den_c(i)+Up(i,k); %Eq 16
        end
    end %End of K-loop
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Update Cluster centers as Eq 16 in paper
    c_old=c;
    for i=1:C
        
        den_c(i)=sum(Up(i,:),2); %Eq 16
        
        if(abs(den_c(i))<eps)
            den_c(i)=eps;
        end
        c(i,:)=num_c(i,:)/((1+alpha)*den_c(i)); %Eq 16
    end
    
    %% Wrong Bias Field Estimate as Eq 19 in Ahmed's paper
    if var
        for k=1:N
            %nomt=sum(repmat(Up(:,k),1,imagesize(3)).*c);
            nomt=sum(c.*Up(:,k));
            dent=sum(Up(:,k));
            if(abs(dent)<eps)
                dent=eps;
            end
            B(k)=Y(k) - nomt/dent;
        end
        if debug
            F(itt)= eq4( reshape(Y,[512 512]),shiftdim(U,1),c,reshape(B,[512 512]),Options,Nind); %Eq 7
            figure(2) %Plot the Objective Function in real time
            plot(F)
            drawnow
        else
            F=0;
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %Low-pass filter Bias-Field, as regularization
        B=imgaussian(reshape(B,imagesize),sigma);
        B=reshape(B,size(Y));
        
    else
        %% Correct bias field estimation
        alphan=alpha/Nr;
        for k=1:N
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Calculate lambda as Eq 18 in paper
            lambda_num=sum(c.*(Up(:,k)+alphan.*(sum(Up(:,Nind{k}),2)))); %Eq 18
            lambda_den=sum(Up(:,k)+alphan.*(sum(Up(:,Nind{k}),2))); %Eq 18
            if(abs(lambda_den)<eps)
                lambda_den=eps;
            end
            lambda(k)=(Y(k)-(lambda_num/lambda_den)); %Eq 18
        end
        lambda_c=sum(lambda)/numel(Y) %Eq 18
        % lambdaN=lambda./-numel(Y);
        clearvars lambda_num lambda_den
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Update Bk as in Eq 17 in paper
        for k=1:numel(Y)
            lambda_num=sum(c.*(Up(:,k)+alphan.*(sum(Up(:,Nind{k}),2)))); %Eq 17
            lambda_den=sum(Up(:,k)+alphan.*(sum(Up(:,Nind{k}),2)));   %Eq 17
            %             lambda_num=sum(c.*(Up(:,k))); %Eq 18
            %             lambda_den=sum(Up(:,k)); %Eq 18
            if(abs(lambda_den)<eps)
                lambda_den=eps;
            end
            B(k)=Y(k)-((lambda_num+lambda_c)/lambda_den); %Eq 17
            %             figure(5) plot(B); drawnow
            % New Bk as shown in my register
            %             B(k)=Y(k)-((numel(Y)*lambda_num -
            %             sum(lambda))/(numel(Y)*lambda_den));
        end
        if debug
            F(itt)= (eq4( reshape(Y,[512 512]),shiftdim(U,1),c,reshape(B,[512 512]),Options,Nind)); %Eq 7
            figure(2) %Plot the objective Function in real time
            plot(F)
            drawnow
        else
            F=0;
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Low-pass filter Bias-Field, as regularization
        B=imgaussian(reshape(B,imagesize),sigma);
        B=reshape(B,size(Y));
        clearvars lambda_num lamda_den
    end
    if debug
        bias_mask(:,:,itt)=reshape(B,size(Y_im)); %% store bias estimate at every iteration
        U2=shiftdim(U,1);
        U2=squeeze(reshape(U2,[imagesize(1:2) C]));
        U_mask(:,:,:,itt)=U2; % store partition matrix at every iteration
        
    else
        bias_mask=0;
        U_mask=0;
    end
    %     if itt>1
    %         stop=F(itt-1)-F(itt);
    %     end
    itt=itt+1; % End of iteration
    
end % end of while loop
% Reshape Final Partition table to image
U=shiftdim(U,1);
% Reshape Final bias field to image
B=reshape(B,imagesize);