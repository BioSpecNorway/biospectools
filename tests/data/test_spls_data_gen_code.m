clear all;
close all;
pkg load tablicious
pkg load statistics
pkg load nan

inp = {'male', 'female', 'male'}
idxs = grp2idx(inp)
cat2bin(idxs)

%% Set path to ako's functions
loadPath='';
%%
DataInFile = 'MouldsNewName.mat';
load(strcat(loadPath,DataInFile))

ZX = selectrow(ZX,1:10);
N =  size(ZX.d,1);

% figure
% plot(str2num(ZX.v),ZX.d);
% set(gca,'XDir','reverse','linewidth',1.5,'FontWeight','Bold');
% set(gcf,'Color',[1 1 1]);
% axis tight;
% xlabel('Wavenumber [cm^-^1]','FontSize',8,'FontWeight','Bold');
% ylabel('Intensity','FontSize',8,'FontWeight','Bold')
% title('Moulds spectra raw');

%% EMSC
[EMSCModel] = make_emsc_modfunc(ZX);
[ZXcor,~,~]=cal_emsc(ZX,EMSCModel,[],[]);

% figure
% plot(str2num(ZXcor.v),ZXcor.d);
% set(gca,'XDir','reverse','linewidth',1.5,'FontWeight','Bold');
% set(gcf,'Color',[1 1 1]);
% axis tight;
% xlabel('Wavenumber [cm^-^1]','FontSize',8,'FontWeight','Bold');
% ylabel('Intensity','FontSize',8,'FontWeight','Bold')
% title('Moulds spectra corrected');

% groups = mat2cell(ZXcor.i(:,8:13),ones(N,1),6);
% groups = categorical(groups);
% ZY.d = dummyvar(groups);
spectra = ZXcor.d;
labels = ZXcor.i(:,8:13);
% wns = ZXcor.v;
groups = grp2idx(labels);
ZY.d = cat2bin(groups);
ZY.i = ZXcor.i;
% ZY.v = cell2mat(categories(groups));
ZY.v = unique(groups);

num_LV = 6;
sparsity = 0.99;
sparsity_arr = sparsity*ones(1,10); % 90%
[Beta,W,P,Q,T,mXin,mYin] = spls(ZXcor,ZY,[],num_LV,sparsity_arr);

W = W.d;
P = P.d;
Q = Q.d;
T = T.d;
B = Beta.d;

save (
  "-mat-binary", "test_spls_data.mat", "spectra", "labels",
  "num_LV", "sparsity", "B", "W", "P", "Q", "T");

%%
% AOpt = 5;
% b.d = Beta.d(:,:,AOpt);  % regression coefficients at AOpt
% b.i = Beta.i;
% b.v = Beta.v;

% figure
% plot(str2num(b.i),b.d);
% set(gca,'XDir','reverse','linewidth',1.5,'FontWeight','Bold');
% set(gcf,'Color',[1 1 1]);
% axis tight;
% xlabel('Wavenumber [cm^-^1]','FontSize',8,'FontWeight','Bold');
% ylabel('Intensity','FontSize',8,'FontWeight','Bold')
%title('sparse regression coefficient');