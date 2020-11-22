%% example of compressed sensing regularization
clear all;

%% 8 channel dataset
load phantom;

% gpu is a lot faster
try
    data = gpuArray(data);
catch
    disp('gpuArray not available');
end

% better scaling
data = data * 1e6;

% create low resolution coil maps
coils = convn(data,ones(11),'same');

% add noise
data = data + 0.001*complex(randn(size(data)),randn(size(data)));

% undersample 4x
data = fft2(data);
mask = false(256,256);
mask(:,1:4:256) = 1;
data = mask.*data;

%% set up matrices
%
% least squares      => D F C    (x)  = y (solve for x)
% compressed sensing => D F C Q' (Qx) = y (solve for Qx which we will force to be sparse)
%
% F = Fourier transform
% C = coil modulation
% D = sampling operator
% Q = wavelet transform
% x = image to recover (ratio)
% y = acquired k-space
%
D = @(arg) mask.*arg;
F = @(arg) fft2(arg);
C = @(arg) coils.*arg;

aD = @(arg) D(arg);
aF = @(arg) ifft2(arg);
aC = @(arg) conj(coils).*arg;

% wavelet transform (Q=fwd Q'=inv)
Q = DWT(size(mask),'db2');

% forward operator (FDCQ')*arg
fwd = @(arg) D(F(C(Q'*reshape(arg,size(mask)))));

% adjoint operator (QC'D'F')*arg
adj = @(arg) sum(Q*aC(aF(aD(arg))),3);

%% normal equation: (adj * fwd) * x = (adj * y)  <==> M x = Y
M = @(arg) reshape(adj(fwd(arg)),[],1);
Y = reshape(adj(data),[],1);

%% without compressed sensing
lambda = 0;
X0 = pcgL1(M,Y,lambda);
X0 = reshape(X0,size(mask));

%% with compressed sensing
lambda = 0.01;
X1 = pcgL1(M,Y,lambda);
X1 = reshape(X1,size(mask));

%% final image with sum of squares modulation

% extract from transform domain
X0 = Q'*X0;
X1 = Q'*X1;

% impart a coil modulation
final_coil_modulation = sqrt(sum(abs(coils).^2,3));
X0 = X0.*final_coil_modulation;
X1 = X1.*final_coil_modulation;

%% display

subplot(1,3,1)
imagesc(sqrt(sum(abs(ifft2(data)).^2,3))); title('undersampled');
subplot(1,3,2)
imagesc(abs(X0)); title('least squares');
subplot(1,3,3)
imagesc(abs(X1)); title(sprintf('CS (lambda=%.2f)',lambda));
