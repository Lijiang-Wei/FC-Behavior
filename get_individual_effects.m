function [flag, m2, SE, Va, Ve, bhat, uhat, Lnew] = get_individual_effects(y, X, W, K, alg, tol, max_iter, verbose)
%
% This function implements morphometricity.
% It fits a linear mixed effects model (LME) using the restricted maximum likelihood (ReML) algorithm, 
% and produces the morphometricity estimate and its standard error. 
%
% If W is provided, the LME is: y = Xb + Wu + e,
% where X is the design (covariate) matrix, b is a vector of fixed effects,
% W is an Nsubj x Nimg standardized matrix containing neuroimaging measurements,
% in which each column has zero mean and unit variance,
% u is a vector of i.i.d. normally distributed random effects: u ~ N(0, Va/Nimg*I),
% and e is a vector of i.i.d. normally distributed noise: e ~ N(0, Ve*I).
% The model is equivalent to y = Xb + a + e,
% where a ~ N(0, Va*K), K = W*W'/N_img.
% That is, W will be used to calculate a linear anatomical similarity matrix (ASM; see below).
%
% Alternatively, if W = [] and K is provided (can be a nonlinear kernel), the LME is: y = Xb + a + e,
% where a ~ N(0, Va*K).
% 
% The covariance structure of the model is: cov(y) = Va*K + Ve*I.
%
% Input -
% y is an Nsubj x 1 vector of phenotypes (trait values)
% X is an Nsubj x Ncov covariate matrix (that contains nuisance variables such as age, sex, site dummy, etc)
% W is an Nsubj x Nimg standardized matrix containing neuroimaging measurements (each column has zero mean and unit variance),
% K is an Nsubj x Nsubj anatomical similarity matrix (ASM)
%    If W is provided, K is computed as W*W'/N_img
%    If K is directly provided, it has to be a symmetric, positive semi-definite matrix with its diagonal selements averaging to 1
%    If K is not non-negative definite, its negative eigenvalues will be set to zero, and a warning will be printed to the command window
% alg is the algorithm for the ReML; default alg = 0
%    alg = 0 - use the average information
%    alg = 1 - use the expected information (Fisher scoring)
%    alg = 2 - use the observed information
% tol is the tolerance for the convergence of the ReML algorithm; default tol = 1e-4
% max_iter is the maximum number of iterations for the ReML algorithm; default max_iter = 100
% verbose = 1 - print to the command window.
%
% Output -
% flag indicates the convergence of the ReML algorithm
%   flag = 1 - the ReML algorithm has converged
%   flag = 0 - the ReML algorithm did not converged
% m2 is the morphometricity estimate
% SE is the standard error of the morphometricity estimate
% Va is the total anatomical/morphological variance
% Ve is the residual variance
% bhat is the ReML estimate for the fixed effects
% uhat is the best linear unbiased predictor (BLUP) for u (if W is provided)
% Lnew is the ReML likelihood when the algorithm is converged

%%% Author: Tian Ge (minor modifications by Mert R. Sabuncu)
%%% Contact: <tge1@mgh.harvard.edu> or <msabuncu@cornell.edu>

%% input check
if nargin < 3
    error('Not enough input arguments')
elseif nargin == 3
    K = []; alg = 0; tol = 1e-4; max_iter = 100; verbose = 0;
elseif nargin == 4
    alg = 0; tol = 1e-4; max_iter = 100; verbose = 0;
elseif nargin == 5
    tol = 1e-4; max_iter = 100; verbose = 0;
elseif nargin == 6
    max_iter = 100; verbose = 0;
elseif nargin == 7
    verbose = 0;
elseif nargin > 8
    error('Too many input arguments')
end
% -----
if isempty(W) && isempty(K) 
    error('Please provide W or K')
elseif ~isempty(W) && isempty(K)
    [~,Nimg] = size(W);
    K = W*W'/Nimg;
elseif ~isempty(W) && ~isempty(K)
    disp('WARNING: K will be computed using W; the input K will be ignored')
    [~,Nimg] = size(W);
    K = W*W'/Nimg;
end
% -----
X = [ones(length(y),1), X];   % add a bias term for X

nan_ind = isnan(y);   % find nan in y
y(nan_ind) = [];
X(nan_ind,:) = [];
if ~isempty(W)
    W(nan_ind,:) = [];
end
K(nan_ind,:) = [];
K(:,nan_ind) = [];

fprintf('In total %d subjects...\n', length(y));
% -----
[U,D] = eig(K);   % calculate the eigenvalues and eigenvectors of the GRM
if min(diag(D)) < 0   % check whether the GRM is non-negative definite
    disp('WARNING: the GRM is not non-negative definite! Negative eigenvalues will be set to zero')
    D(D<0) = 0;   % set negative eigenvalues to zero
    K = U*D/U;   % reconstruct the GRM
end
%% derived quantities
Nsubj = length(y);   % calculate the total number of subjects
Vp = var(y);   % calculate the phenotypic variance
%% initialization
Va = Vp/2; Ve = Vp/2;   % initialize the anatomical variance and residual variance
V = Va*K+Ve*eye(Nsubj);   % initialize the covariance matrix
P = (eye(Nsubj)-(V\X)/(X'/V*X)*X')/V;   % initialize the projection matrix
%% EM algorithm
if verbose
    disp('---------- EM algorithm ----------')
end
% use the expectation maximization (EM) algorithm as an initial update
Va = (Va^2*y'*P*K*P*y+trace(Va*eye(Nsubj)-Va^2*P*K))/Nsubj;   % update the anatomical variance
Ve = (Ve^2*y'*P*P*y+trace(Ve*eye(Nsubj)-Ve^2*P))/Nsubj;   % update the residual variance

% set negative estimates of the variance component parameters to Vp*1e-6
if Va < 0; Va = 10^(-6)*Vp; end 
if Ve < 0; Ve = 10^(-6)*Vp; end

V = Va*K+Ve*eye(Nsubj);   % update the covariance matrix
P = (eye(Nsubj)-(V\X)/(X'/V*X)*X')/V;   % update the projection matrix

E = eig(V); logdetV = sum(log(E+eps));   % calculate the log determinant of the covariance matrix

Lold = inf; Lnew = -1/2*logdetV-1/2*log(det(X'/V*X))-1/2*y'*P*y;   % initialize the ReML likelihood
%% ReML
if verbose
    disp('---------- ReML iterations ----------')
end

iter = 0;   % initialize the total number of iterations
while abs(Lnew-Lold)>=tol && iter<max_iter   % criteria of termination    
    % new iteration
    iter = iter+1; Lold = Lnew;
    if verbose
        disp(['---------- ReML Iteration-', num2str(iter), ' ----------'])
    end
    
    % update the first-order derivative of the ReML likelihood
    Sg = -1/2*trace(P*K)+1/2*y'*P*K*P*y;   % score equation of the anatomical variance
    Se = -1/2*trace(P)+1/2*y'*P*P*y;   % score equation of the residual variance
    S = [Sg;Se];   % construct the score vector
    
    % update the information matrix
    if alg == 0
        Igg = 1/2*y'*P*K*P*K*P*y; Ige = 1/2*y'*P*K*P*P*y; Iee = 1/2*y'*P*P*P*y;   % average information
    elseif alg == 1
        Igg = 1/2*trace(P*K*P*K); Ige = 1/2*trace(P*K*P); Iee = 1/2*trace(P*P);   % expected information
    elseif alg == 2
        Igg = -1/2*trace(P*K*P*K)+y'*P*K*P*K*P*y; Ige = -1/2*trace(P*K*P)+y'*P*K*P*P*y; Iee = -1/2*trace(P*P)+y'*P*P*P*y;   % observed information
    end
    I = [Igg,Ige;Ige,Iee];   % construct the information matrix
    
    T = [Va;Ve]+I\S; Va = T(1); Ve = T(2);   % update the variance component parameters
    
    % set negative estimates of the variance component parameters to Vp*1e-6
    if Va < 0; Va = 10^(-6)*Vp; end 
    if Ve < 0; Ve = 10^(-6)*Vp; end
    
    V = Va*K+Ve*eye(Nsubj);   % update the covariance matrix
    P = (eye(Nsubj)-(V\X)/(X'/V*X)*X')/V;   % update the projection matrix
    
    E = eig(V); logdetV = sum(log(E+eps));   % calculate the log determinant of the covariance matrix
    
    Lnew = -1/2*logdetV-1/2*log(det(X'/V*X))-1/2*y'*P*y;   % update the ReML likelihood
end
%% morphometricity estimate and standard error
m2 = Va/(Va+Ve);   % morphometricity estimate

% update the information matrix at the final estimates
if alg == 0
    Igg = 1/2*y'*P*K*P*K*P*y; Ige = 1/2*y'*P*K*P*P*y; Iee = 1/2*y'*P*P*P*y;   % average information
elseif alg == 1
    Igg = 1/2*trace(P*K*P*K); Ige = 1/2*trace(P*K*P); Iee = 1/2*trace(P*P);   % expected information
elseif alg == 2
    Igg = -1/2*trace(P*K*P*K)+y'*P*K*P*K*P*y; Ige = -1/2*trace(P*K*P)+y'*P*K*P*P*y; Iee = -1/2*trace(P*P)+y'*P*P*P*y;   % observed information
end
I = [Igg,Ige;Ige,Iee];   % construct the score vector and the information matrix

invI = inv(I);   % inverse of the information matrix
SE = sqrt((m2/Va)^2*((1-m2)^2*invI(1,1)-2*(1-m2)*m2*invI(1,2)+m2^2*invI(2,2)));   % standard error estimate
%% BLUP of u
bhat = (X'/V*X)\(X'/V*y);   % ReML estimate for the fixed effects
if isempty(W)
    uhat = NaN;
else
    uhat = Va*W'/V*(y-X*bhat)/Nimg;   % BLUP for u
end
%% diagnosis of convergence
if iter == max_iter && abs(Lnew-Lold)>=tol
    flag = 0;
else
    flag = 1;
end
%%