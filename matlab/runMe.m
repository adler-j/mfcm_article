clear all
close all
clc
% This code calls the mfcm2 function which estimates the bias field
% And plots the results.
% Written by Awais Ashfaq - KTH 2016
param_setting; % Set Fuzzy parameters
flag=0; % Change to 1 if you want to estimate bias field using AHMED'S approach [1].
debug=0; % set to 1 if need to visualize bias mask and partition matrix at every iteration.
I0= mat2gray(im2double(load ('recon_scatter_0.05_noise_0.01.txt'))); % Load Input Image
Y=imgaussian(I0,0.5); % Apply gaussian smoothing to remove additive noise.
c=multithresh(Y,3)'; % Estimate cluster centers using Otsu's Method
c(1)=0;
% Begin Fuzzy clustering
% Y is the filtered Input Image
% c is the estimated cluster centers
% param is a struct carrying Fuzzy Parameters
% Flag determines the method used for Fuzzy
% debug provides additional information (bias bask, partition matrix) for every iteration
[B,U,bias_mask,U_mask,C,F]=mfcm2(Y,c,param,flag,debug);
out=Y-B; % Subtract the bias field from the filtered input Image
U2=squeeze(reshape(U,[size(Y) numel(c)]));
% Show results
figure(1),
subplot(2,2,1), imshow(Y), title('Input image');
subplot(2,2,2), imshow(U2,[]), title('Partition matrix');
subplot(2,2,3), imshow(B,[]), title('Estimated biasfield');
subplot(2,2,4), imshow(out,[]), title('Corrected image');
if debug
    figure(3)
    imshow3d(bias_mask,[]);
    figure(4)
    montage(U_mask);
    set(gca,'clim',[0 4])
end
%% Compare segmentation results
% c=multithresh(Y,3);
% out1=imquantize(Y,c);
% out1=changem(out1,[0 0.5 1 1],[1 2 3 4]);
%
% c=multithresh(out,3);
% out2=imquantize(out,c);
% out2=changem(out2,[0 0.5 1 1],[1 2 3 4]);
% figure(2)
% subplot(1,2,1)
% imshow(out1)
% title('Thresholding before Bias correction')
% subplot(1,2,2)
% imshow(out2)
%
% title('Thresholding after Bias correction')


% 1. Ahmed MN, Yamany SM, Mohamed N, Farag AA, Moriarty T. A modified fuzzy c-means algorithm for bias field estimation and segmentation of MRI data. IEEE transactions on medical imaging. 2002;21(3):193-9
% Available at: http://www.ncbi.nlm.nih.gov/pubmed/11989844