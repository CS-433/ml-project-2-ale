clc
clear
close all

subject         = 1;                         % NOTE 1
session         = 1;                         % NOTE 2
band            = 1;                         % NOTE 3
signalformat    = 2;                         % NOTE 4 
WhichPenalty    = 'log';                     % NOTE 5
WhichDist       = 'gspbox';                  % NOTE 6 
alphas          = [0.05 0.1 0.2 0.5 0.75 1]; % NOTE 7 
betas           = [0.05 0.1 0.2 0.5 0.75 1]; % NOTE 8
epochsets       = {                          % NOTE 9
    1
    1:5
    randperm(20,5)
    []
    };
do_downsample   = 0;                         % NOTE 10
dsample_factor  = 5;                         % NOTE 11

%% Addpaths & initiate GSPBOX.

addpath 'utils'
addpath 'utils/gspbox'
addpath 'utils/HCP_info'

gsp_start

FirstGSPBoxUse = 0;
if FirstGSPBoxUse
    gsp_make %#ok<*UNRCH>
    
    gsp_install
    
    gsp_install_unlocbox_hb
end

%% Load a sample file.

dir_data = '../data'; 
%dir_data = '/media/miplab-nas2/HCP-Data/HCP_MEG/DATA_MAT_bandpass-envelope/SourceRecon_a2009_Centroids';

d = load('MEG84_subjects_ID.mat');
IDs = d.IDs;

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

signalformats = {
    'bp'
    'env'
    };

n_files = getnames(dir_data);

ID      = IDs{subject};
sess    = sessions{session};
sigform = signalformats{signalformat};

tag = sprintf('%s_%s_%s',ID,sess,bands{band});

for k=1:length(n_files)
    if contains(n_files{k},tag)
        f_load = fullfile(dir_data,n_files{k});
        break;
    end
end

d = load(f_load);
DATA = d.ts;

XX = DATA.(sigform);

if do_downsample
    DSF = dsample_factor;
    XX = XX(:,1:DSF:end,:)
end

Natls = size(XX,1);
Nfrms = size(XX,2);
Nephs = size(XX,3);
Nsets = length(epochsets);
Nalph = length(alphas);
Nbeta = length(betas);

switch WhichPenalty
    case 'l2'
        W = zeros(Natls,Natls,Nalph,Nsets);
    case 'log'
        W = zeros(Natls,Natls,Nbeta,Nsets);
end

%% Learn graphs.

for iSet=1:Nsets
    
    I = epochsets{iSet};
    if isempty(I)
       I = 1:Nephs;
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
    
    switch WhichPenalty
        case 'log'
            for k=1:Nbeta
                W(:,:,k,iSet) = gsp_learn_graph_log_degrees(Z,1,betas(k));
            end
        case 'l2'
            for k=1:Nalph
                W(:,:,k,iSet) = gsp_learn_graph_l2_degrees(Z,alphas(k));
            end
    end
end

%% Inspect graphs.
close all;
for iSet=1:Nsets
    hf = figure(iSet);
    set(hf,...
        'Position',[10 50 1600 600],...
        'Name',sprintf('frame-set: %d',iSet));
    k=0;
    for iParam=[1,3]
        k=k+1;
        subplot(1,2,k);
        w = W(:,:,iParam,iSet);
        imagesc(w);
        switch WhichPenalty
            case 'log'
                title(sprintf('ID: %s . sess: %s . sigform: %s . band: %s . beta: %.02f',ID, sess(1), sigform, bands{band}, betas(iParam)));
            case 'l2'
                title(sprintf('ID: %s . sess: %s . sigform: %s . band: %s . beta: %.02f',ID, sess(1), sigform, bands{band}, alphas(iParam)));
        end
        colormap hot;
        colorbar
        xlabel('brain regions');
        ylabel('brain regions');
        axis image
    end
end

%==========================================================================
function n_files=getnames(d_data)
d = dir(d_data);
dname = {d.name};
fn = @(x) contains(...
    x,'Restin_rmegpreproc_bandpass-envelop')...
    && contains(x,'.mat');
n_files = dname(cellfun(fn,dname));
end

%% NOTES.

% NOTE 1
% Subject number; an integer in [1 84].
%
% NOTE 2
% Session number; 1 or 2. There are two resting state sessions per subject,
% per frequency band. The data for each session, and each frequency band
% consists of multiple epochs, each epoch of length 12 second (6108
% samples). Given that there are two sessions per subject, one idea is to
% use data from one session for training and those of the other for
% testing. A second startegy is that you pull together data from session 1
% and 2, and then, you split those into two chunks, using one chunk for
% training and another for testing. If you get good performance of the
% first trategy, that is prefered. If that doesn't work, then also try
% strategy two. Or just report on both strategies.  For instance, in
% Amico2016, which is a fingerprinting study on HCP fMRI data, they mix
% data from two sessions (LR and RL sessions) for their test retest
% analysis.
%
% NOTE 3
% frequency band number; an integer in [1 5]. 
% The MEG data we temporaly decomposed into five canonical frequency bands:
% delta: [1 4] Hz
% theta: [4 8] Hz
% alpha: [8 13] Hz 
% beta : [13 30] Hz 
% gamma: [30 48] Hz
%
% NOTE 4
% Signal format to use; 1 (bp) or 2 (env). 
% Time courses are provided for each frequency band, in two formats: <bp>
% and <env>; <bp> is the bandpassed filtered signal whereas <env> is the
% envelope of that signal.
%
% NOTE 5
% There are two options here: 'log' or 'l2'. For further details, see the
% related theory in the paper Kalofolias2016 and the related note below,
% under literature. 
%
% NOTE 6 
% Distance measure to use in the learning. 'euclidean' or 'gspbox' or you
% may explore other distance measures as you see appropriate to improve the
% performance.
%
% NOTE 7
% If using the l2 penality, there is one free parameter in the algorithm,
% alpha.  
%
% NOTE 8
% If using the log penality, there are two parameters in the algorithm,
% alpha and beta, but it can be shown that it is sufficinet to set alpha to
% a fixed value of and then do a grid search on beta as these two
% parameters are linked. 
%
% epfl-lts2.github.io/gspbox-html/doc/demos/gsp_demo_learn_graph.html
% 
% Instead of setting the parameters and exploring the performance via grid
% search, it may be beneficial to set the parameters automatically by
% specifying a desired sparsity; that is, you do your analysis on a desired
% range of sparsities, and check the performance in relation to sparity of
% the graphs; see: 
%
% epfl-lts2.github.io/gspbox-html/doc/demos/gsp_demo_learn_graph_large.html
%
% However, if the imposed sparsity i at the level of nodes (i.e. fixing the
% number of edges allocated to each node) then that is not suitable for
% us since this is a strong assumption (and invalid) to consider brain
% region having same number of connections (nodal density). If [Kalofolias,
% Perraudin 2019] has this assumption, disregard that approach, and stick
% with varying the learning paarmeters.
%
% For both NOTE 7 and 8, consider:
% 
% [?] How does the performance vary as a function of the chosen learning
% parameter? Is there a sweet spot in the parameter space at which we see
% optimal performance? 
%
% NOTE 9
% The data for each session, each band is split into multiple epochs. For
% both signalformats (i.e., <bp> or <env>), an array of dimension N×T×E,
% where N=148 is the number of brain regions, T is the number of time
% points within and epoch, and E is the number of epochs. N is identical
% throughout the data, whereas T and E may differ across subjects.
%
% You should explore, in a systematic way, different options when learning
% the graphs in terms of number of epochs used, and how they are seletced.
% By number of samples I mean the number of frames from the session. Here I
% have just specied some ad-doc frame sets for this demo. Explore, and try
% to answer:
%
% [?] Does the performance increase as you icrease the number frames used to
% learned the graphs? 
%
% [?] Is there a minimum number of frames after which the performance
% reached a plateau?
%
% [?] Does it matter whether a set of "consecutive frames" or "random
% frames" are used from the session?
%
% NOTE 10
% Should the data be downsampled (1) or not (0)?
% Note that the data are at a high sampling rate; ~509 Hz. However, we have
% performed bandpass filtering, dampening frequencies above 48, at the
% least. Each epoch is 12 seconds long (6108 samples), thus, to reduce the
% dimention, we can downsample the data to certain degree, without loosing
% much info since high frequencies have been filtered out. Plot some of the
% time courses to get a good feeling of the data. 
%
% If you downample by a factor of 5, the sampling rate becomes ~100 Hz,
% half of which (known as the Nyquist frequency) is above the highest
% frequency present in the data (48 Hz), and thus, you are not loosing
% temporal information. But if you see that you need more samples for
% learning, either uing a smaller downsampling factor (2, 3 ..) or even
% skip downsampling.
%
% NOTE 11
% Downsampling factor; an integer in [2 5]. Note that for the lower
% frequency bands you can even use a higher downsampling rate; for
% instance, for the delta band you can even downsample by a factor of 50,
% thus, leading to a Nyquist frequency of 5 Hz that is above 4 Hz. But if
% you want to be consistent across frequency band in terms of the number of
% samples you want to use, an upper limit of 5 is a safe choice since then
% it does not destroy information in the gamma band.
%
% NOTE 12 
% All the data, across subjects and sessions, are stored in this folder, in
% total 840 files. Each file is a structure with multiple fields. In
% particular, the fields <bp> and <env> store the data that you want to
% use, with the format: brain regions x time points x epochs.
%
% -----------
% Literature.
%
% Sareen2021: in this work, Ekansh and colleaues use the same set of MEG
% data data you are using, and explore its figerprinting power. In
% particular, they explore the fingerprinting power of the data via
% functional connectivity (FC) matrices, constructing by computing the
% correlation between time courses of pairs of brain regions; and they
% explore multiple flavors of FC matrices, encoding amplitude and phase. In
% your setting, you are performing the same task, but not using FC matrices
% derived from the data, but adjacencuy matricies of learned graphs.
% Moreover, you want to develope a suitable classifier, whereas in
% Sareen2021, they assess the fingerprinting power via inter-class
% correlation (ICC) matirces, and associated measures derived from them,
% I_{self} and I_{others}. And the ICC is build via correaltion, seeing
% which matrix best matche an input matrix. But you would want to build a
% more sophisticated classifer, based on ML methods you have learned from
% your course, e.g., SVMs, Decision Trees, Neural Nets, etc.  
% 
% Amico2016: this paper is good reference, which helps you better
% understand the idea of fingerprinting using a similarity matrix build
% from brain functional actvity data. In the paper they use FC matrices, a
% in Sareen2021, and in particular, they use a PCA approach to reconstruct
% FC matrices. You should consider exploring the same PCA approach also on
% your learned graphs; it's a simple but effective approach, and good to
% showcase for your course, to tick off more ML techniques. 
% 
% Gao2021: this paper i one of the very few papers (maybe only major one so
% far) that looks into graph learning on brain imaging data. They use the
% same graph learning strategy that is used in this demo script, i.e., the
% method in Kalofolias2016. read an get inpiration from. The anlysis is on
% fMRI data, but there are still many similarities, ideas to get
% inspiration from. As in the other works, they use the correlationa and
% matching strategy to assess fingerprinting power; see page 4, paragraph
% "Fingerprinting analysis...". You would want to improve that methodology,
% learning a suitable classifier. 
%
% Kalofolias2016: provides the theory for the graph learning method used
% below, and in Gao2021. If in case you manage to explore other potential
% graph learning approches, like a deep learning one that one of you
% mentioned, that will indeed be a big bonus, which, if it works, making
% your work very novel; for instance, check out the following newly relased
% library, which implememts multiple different methods:
% 
% https://github.com/maxwass/pyGSL
%
% -------------------------------------
% Some ideas to think about and explore: 
% 
% [1] For an atlas with fine-grained parcels, e.g. the one with 600 or 800
% regions, you most likley cannot learn a suitable graph given the limited
% number of frames we have per session, 176 to 1200 per session; there are
% in total 8048 frames per subject ([176+253+284+232+274+405+1200+1200]*2);
% see <sessions_length.mat>. As such, you might want to consider using a
% strategy to learn a graph, for a given subject, using the ensemble set of
% frames we have for that subject across sessions. You then want to assess
% whether this graph, learned from data from across all sessions (task and
% rest, or say just task), can be used to predict the identity of a given
% subject from a graph learned from a test set. In doing so, for the task
% data, this is in paryicular more meanigful for setting=2, since in the
% preprocessing performed in that setting, the effect of different
% experimental paradigms that subjects were performing during tasks has
% been regresed out from the data, and as such, the data are more similar,
% becoming to some degree like rest data.
% 
% [2] Related to [1], given that we have limited data per subject to learn
% large graphs, consider implementing a "data augmentation" scheme and see
% if this helps.
% 
% [3] As in Amico2016, I think you might benefit from learning a
% representation space from the entire set of learned graphs, across
% subjects, and then using the eigemodes of that template graph, to
% decompose a given input graph, and, then reconstruct it using a suitable
% subset of the eigenmodes, and then perform fingerprinting. The "suitable
% subset" is something that you figure out during training. In this way, by
% tranforming an input graph to a econdary graph, you remove much of the
% variance that is common between subjects, only retaining the variance
% that relates to subject-specific attributes. You may also think of using
% strategies other than PCA.
