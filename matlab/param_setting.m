% This file stores the parameters of the FCM function
% Written by Awais Ashfaq - KTH 2016
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% alpha controls the contribution of neighborhood pixel values
% when estimating bias field. A higher alpha is preferred for a more noisy
% input image
param.alpha=1;  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sigma controls the gaussian smoothing of the bias field.
param.sigma=25;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% m conrols the amount of fuzziness in the membership or partition matrix.
% A higher m gives a more fuzzy partition matrix
param.m=2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% epsilon is the stopping criterion as defined in Eq 19 of the paper
param.epsilon=1e-5;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% maxit is the maximum number of iterations performed prior to termination.
% It also serves as a stopping criterion incase Eq 19 is never satisfied
param.maxit=40;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% win is the neighborhood window size. win = 3 means a window of 3 x 3
% around the pixel of interest. This means a neighborhood size of 8 where
% the center pixel of interest (center pixel) is excluded
param.win=3; % Must be odd % The current function only supports win=3