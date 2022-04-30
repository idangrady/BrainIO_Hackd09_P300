%% Load all data, segmenting into each trial

dat1 = load("S1.mat");
dat2 =load("S2.mat");
dat3 =load("S3.mat");
dat4 =load("S4.mat");
dat5 =load("S5.mat");

dat = {dat1,dat2,dat3,dat4,dat5};


%% CAR
raw = cell(5,2);

for i = 1:5
    mfile = matfile( sprintf('S%i.mat', i));
    raw{i,1} = mfile.y;
    raw{i,2} = mfile.trig;
end

for i = 1:5
    CAR{i,1} = median(raw{i,1}, 2);
    raw{i,1} = raw{i,1} - CAR{i,1};
end


%% band pass filter parameters, butterworth, 8 degree, 0.5-35Hz
SOS = [
    1.0000         0   -1.0000    1.0000   -1.9905    0.9906
    1.0000         0   -1.0000    1.0000   -0.9937    0.5535
    1.0000         0   -1.0000    1.0000   -1.9765    0.9766
    1.0000         0   -1.0000    1.0000   -0.7666    0.1776];
G = [0.3696
    0.3696
    0.3218
    0.3218
    1.0000];

[a,b]=butter(8,18/(250/2),'low'); %%%%%%%%%%% added

%% filtering
paddingLength = 500;
for i = 1:5
    temp = [ones(paddingLength,8).*raw{i,1}(1,:); raw{i,1}; ones(paddingLength,8).*raw{i,1}(end,:)];
    temp = filtfilt( SOS, G, temp);
    raw_f{i,1} = temp( paddingLength+1:size(temp,1)-paddingLength, :);
    raw_f{i,1} = filtfilt(a,b,raw_f{i,1}); %%%%%%%%%%% added
end

%% replace the data(y) in .dat
for i = 1:5
    dat{i}.y = raw_f{i,1};
end

%%
findat = {};
fintar = {};
for indexs = 1:numel(dat)
    gbp = dat{indexs};
    A = gbp.trig;
    A1 = gbp.y;
    
    %     b = [1, -1];
    %     c = ismember(A, b);
    % Extract the elements of a at those indexes.
    %     indexes = find(c);
    
    start_index  = find(A);
    end_index = start_index + 125; % 0-500ms
    n  = numel(start_index);
    Result = cell(1, n);
    for k = 1:n
        dataA1= A1(start_index(k):end_index(k)-1, :);
        Result{k} = resample(dataA1,1,5); %%%%%%%%%%% added
    end
    
    % adding segmented trials to data
    gbp.dat = Result;
    gbp.trig = nonzeros(gbp.trig);
    gbp.y = [];
    dat{indexs} = gbp;
    % findat.append(Result);
    %fintar.append()
    
end

save('data_prepro_resampled.mat', 'dat');

% % %% check the results
% % subject = 2;
% % ch = 5;
% % Data = reshape( dat{1,subject}.dat, 1,1,1200);
% % Data = cell2mat(Data);
% % 
% % posDataMean = mean( Data(:,:,dat{1,subject}.trig==1),3);
% % pos = find(dat{1,subject}.trig==1);
% % data2plot = dat{1,subject}.dat(1,pos);
% % data2plot = cell2mat( reshape( data2plot, 1,1,numel(data2plot)));
% % figure;
% % hold on
% % plot( [1:125]/250, squeeze(data2plot(:,ch, 1:100)));
% % plot( [1:125]/250, posDataMean(:,ch), 'LineWidth', 3, 'Color', 'k');
% % xlabel('time (s)')
% % ylabel('amp')
% % grid on
% % title( sprintf('positive trig, subject %i, ch %i', subject, ch))
% % %% --------
% % negDataMean = mean( Data(:,:,dat{1,subject}.trig==-1),3);
% % neg = find(dat{1,subject}.trig==-1);
% % data2plot = dat{1,subject}.dat(1,neg);
% % data2plot = cell2mat( reshape( data2plot, 1,1,numel(data2plot)));
% % figure;
% % hold on
% % plot( [1:125]/250, squeeze(data2plot(:,ch, 1:100)));
% % plot( [1:125]/250, negDataMean(:,ch), 'LineWidth', 3, 'Color', 'k');
% % xlabel('time (s)')
% % ylabel('amp')
% % grid on
% % title('negaitive trig')
% % title( sprintf('positive trig, subject %i, ch %i', subject, ch))