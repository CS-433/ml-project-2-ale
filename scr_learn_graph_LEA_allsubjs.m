clc
clear
close all

%-Settings.
%--------------------------------------------------------------------------
WhichSubjects   = 1:84;
alphas          = 0.05:0.05:1;
betas           = 0.05:0.05:1;

%dir_ALE = '/media/miplab-nas2/Data3/Hamid_ML4Science_ALE';
%dir_save = fullfile(dir_ALE,'MATLAB/learned_graphs');

%-Prepare Directory.
%--------------------------------------------------------------------------
dir_ALE = '/media/miplab-nas2/Code/Hamid_ML4Science_ALE';
dir_save = '/media/miplab-nas2/Data3/Hamid_ML4Science_ALE/MATLAB/learned_graphs';

dir_script = fileparts(mfilename('fullpath'));

%d1 = fullfile(fileparts(dir_script),'utils');
d2 = fullfile(fileparts(fileparts(dir_script)),'utils');
d1= '/media/miplab-nas2/Code/Hamid_ML4Science_ALE/utils';

if exist(d1,'dir')
    dir_utils = d1;
elseif exist(d2,'dir')
    dir_utils = d2;
else
    error('Cannot locate utils folder.')
end

addpath(dir_utils);
addpath(fullfile(dir_utils,'gspbox'));
addpath(fullfile(dir_utils,'HCP_info'));

% initiate GSPBOX
gsp_start
FirstGSPBoxUse = 0;
if FirstGSPBoxUse
    gsp_make %#ok<*UNRCH>
    
    gsp_install
    
    gsp_install_unlocbox_hb
end

d = '/media/miplab-nas2/HCP-Data/HCP_MEG/DATA_MAT_bandpass-envelope';
dir_data = fullfile(d,'SourceRecon_a2009_Centroids');

d = load('MEG84_subjects_ID.mat');
IDs = d.IDs(WhichSubjects);

Nsubjs = length(IDs);

sessions = {
    '3-Restin_rmegpreproc_bandpass-envelop'
    '4-Restin_rmegpreproc_bandpass-envelop'
    };

bands = {
    'delta'
    'theta'
    'alpha'
    'beta'
    'gamma'
    };

WhichDist = 'gspbox';

signalformats = {
    'bp'
    'env'
    };
signalformat  = 2;
sigform       = signalformats{signalformat};

n_files = getnames(dir_data);

do_downsample  = 0;
if do_downsample
    dsample_factor = 5;
else
    dsample_factor = 0;
end

assert(not(isempty(dir_save)));
if ~exist(dir_save,'dir')
    mkdir(dir_save);
end

runinfo = struct;
runinfo.dir_data  = dir_data;
runinfo.dir_save  = dir_save;
runinfo.sessions  = sessions;
runinfo.bands     = bands;
runinfo.alphas    = alphas;
runinfo.betas     = betas;
runinfo.sigform   = sigform;
runinfo.WhichDist = WhichDist;
runinfo.n_files   = n_files;
runinfo.do_downsample = do_downsample;
runinfo.dsample_factor = dsample_factor;

%-Learn graphs.
%--------------------------------------------------------------------------
tt = tic;
if length(IDs)==1
    
    ID = IDs{1};
    
    runsubj(ID,runinfo);
    
    fprintf(sprintf('Subject done. [%s]',ID));
else
    if Nsubjs<21
        Np = Nsubjs;
    else
        Np = 21; 
        % 84/21 = 4 subjs per core; 
        % there r more than 21 cores, but good to be considerate of others.
    end
    hb_parfor_prepare(Np);
    parfor iSubj=1:Nsubjs
        
        ID = IDs{iSubj};
        
        runsubj(ID,runinfo);
        
        fprintf(sprintf('\nSubject done. [%s] \n',ID));
    end
end
disp('Script done.');
disp(' ');
toc(tt)

%==========================================================================
function n_files=getnames(d_data)
d = dir(d_data);
dname = {d.name};
fn = @(x) contains(...
    x,'Restin_rmegpreproc_bandpass-envelop')...
    && contains(x,'.mat');
n_files = dname(cellfun(fn,dname));
end

%==========================================================================
function runsubj(ID,runinfo)

dir_data       = runinfo.dir_data;
dir_save       = runinfo.dir_save;
sessions       = runinfo.sessions;
bands          = runinfo.bands;
alphas         = runinfo.alphas;
betas          = runinfo.betas;
sigform        = runinfo.sigform;
WhichDist      = runinfo.WhichDist;
n_files        = runinfo.n_files;
do_downsample  = runinfo.do_downsample;
dsample_factor = runinfo.dsample_factor;

Nsess  = length(sessions);
Nbands = length(bands);

for iSess=1:Nsess
    
    sess = sessions{iSess};
    
    for iBand=1:Nbands
        
        band = bands{iBand};
        
        tag = sprintf('%s_%s_%s',ID,sess,band);
        
        for k=1:length(n_files)
            if contains(n_files{k},tag)
                f_load = fullfile(dir_data,n_files{k});
                break;
            end
        end
        
        d = load(f_load);
        d = d.ts;
        XX = d.(sigform);
        
        if do_downsample
            DSF = dsample_factor;
            XX = XX(:,1:DSF:end,:);
        end
        
        if iBand==1
            Natls  = size(XX,1);
            Nfrms  = size(XX,2);
            Nephs  = size(XX,3);
            Nsets  = Nephs+1;   % 1 graph learned per epoch + 1 graph learned using data from all epochs
            Nalph  = length(alphas);
            Nbeta  = length(betas);
            
            W_l2  = zeros(Natls,Natls,Nalph,Nsets);
            W_log = zeros(Natls,Natls,Nbeta,Nsets);
        end
        
        for iSet=1:Nsets
            
            if iSet==Nsets
                I = 1:Nephs;
            else
                I = iSet;
            end
            
            X = XX(:,:,I);
            
            if length(I)>1
                X = reshape(X,Natls,Nfrms*size(X,3));
            end
            
            X = X'; % observations/frames in ROWS not columns for code below
            
            switch WhichDist
                case 'gspbox'
                    Z = gsp_distanz(X).^2;
                case 'euclidean'
                    d = pdist(X','euclidean');
                    Z = squareform(d);
            end
            Z = Z./max(Z(:));
            
            for iPar=1:Nalph
                W_l2(:,:,iPar,iSet) = gsp_learn_graph_l2_degrees(Z,alphas(iPar));
            end
            
            for iPar=1:Nbeta
                W_log(:,:,iPar,iSet) = gsp_learn_graph_log_degrees(Z,1,betas(iPar));
            end
        end
        
        RESULTS = struct;
        RESULTS.ID      = ID;
        RESULTS.band    = band;
        RESULTS.session = sess;
        RESULTS.alphas  = alphas;
        RESULTS.betas   = betas;
        RESULTS.WhichDistMeasureForGraphLearning = WhichDist;
        RESULTS.WhichEEGSignalFormat             = sigform;
        RESULTS.W_l2         = W_l2;
        RESULTS.W_log        = W_log;
        RESULTS.W_l2_size    = size(W_l2);
        RESULTS.W_log_size   = size(W_log);
        RESULTS.W_l2_format  = 'regions x regions x alphas x epochs';
        RESULTS.W_log_format = 'regions x regions x betas  x epochs';
        RESULTS.note1         = 'size of last array dimension (epochs) is one more than number of epochs, since the last one is the graph learned using the entire set of epochs.';
        
        % save results
        if do_downsample
            n = sprintf('%s.%s.%s.sigformat_%s.distance_%s.alphas_%d.betas_%d.downsampleFactor_%d.mat',ID,sess(1:8),band,sigform,WhichDist,Nalph,Nbeta,dsample_factor);
        else
            n = sprintf('%s.%s.%s.sigformat_%s.distance_%s.alphas_%d.betas_%d.noDownsampling.mat',     ID,sess(1:8),band,sigform,WhichDist,Nalph,Nbeta);
        end
        dir_saveID = fullfile(dir_save,ID);
        if ~exist(dir_saveID,'dir')
            mkdir(dir_saveID);
        end
        f = fullfile(dir_saveID,n);
        save(f,'RESULTS');
        
        fprintf('\nDone. [ID: %s - Session: %d - Band: %d] \n\n',ID,iSess,iBand);
    end
end
end

%==========================================================================
function p = hb_parfor_prepare(L,varargin)
%HB_PARFOR_PREPARE initiates a parallel pool. The number of workers will be
%the maximum number of workers available on the local profile, unless L
%(length of for loop) is smaller than that. In the latter case, the size of
%the pool will be equal to L. If second input is provided, i.e., desired
%size of pool, the size of the initiated pool will be equal or smaller to
%that value.
%
% Inputs: 
%   L: length of for loop to run in parallel.
%   varargin{1}: desired parpool size.
%
% Hamid Behjat

if nargin<2
    Np = [];
else
    Np = varargin{1};
end

d = parcluster('local'); 
Nmax = d.NumWorkers; % max possible # of workers

if isempty(Np)
    Np = Nmax;
else
    if Np>Nmax
        s = ['[HB] desired parpool size ',...
            'larger than max possible pool size;',...
            ' a smaller pool is initiated.'];
        warning(s);
        Np = Nmax;
    end
end

if L<Np
    Np=L;
end

p = gcp('nocreate');
if isempty(p)
    p = parpool(Np);
elseif p.NumWorkers~=Np
    delete(p);
    p = parpool(Np);
end
end
