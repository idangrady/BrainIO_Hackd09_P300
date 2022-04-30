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
    CAR{i,1} = median( raw{i,1}, 2);
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

%% filtering
paddingLength = 500;
for i = 1:5
    temp = [ones(paddingLength,8).*raw{i,1}(1,:); raw{i,1}; ones(paddingLength,8).*raw{i,1}(end,:)];
    temp = filtfilt( SOS, G, temp);
    raw_f{i,1} = temp( paddingLength+1:size(temp,1)-paddingLength, :);
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
        Result{k} = A1(start_index(k):end_index(k)-1, :); 
    end

    % adding segmented trials to data
    gbp.dat = Result;
    gbp.trig = nonzeros(gbp.trig);
    gbp.y = [];
    dat{indexs} = gbp; 
   % findat.append(Result);
    %fintar.append()

end

save('data_prepro.mat', 'dat', '-v7.3');

%% check the results
subject = 1;
ch = 7;
Data = reshape( dat{1,subject}.dat, 1,1,1200);
Data = cell2mat(Data);

posDataMean = mean( Data(:,:,dat{1,subject}.trig==1),3);
pos = find(dat{1,subject}.trig==1);
data2plot = dat{1,subject}.dat(1,pos);
data2plot = cell2mat( reshape( data2plot, 1,1,numel(data2plot)));

figure;
hold on
plot( [1:125]/250, squeeze(data2plot(:,ch, 1:100)));
plot( [1:125]/250, posDataMean(:,ch), 'LineWidth', 3, 'Color', 'k');
xlabel('time (s)')
ylabel('amp')
grid on
title( sprintf('positive trig, subject %i, ch %i', subject, ch))
%% --------
negDataMean = mean( Data(:,:,dat{1,subject}.trig==-1),3);
neg = find(dat{1,subject}.trig==-1);
data2plot = dat{1,subject}.dat(1,neg);
data2plot = cell2mat( reshape( data2plot, 1,1,numel(data2plot)));

figure;
hold on
plot( [1:125]/250, squeeze(data2plot(:,ch, 1:100)));
plot( [1:125]/250, negDataMean(:,ch), 'LineWidth', 3, 'Color', 'k');
xlabel('time (s)')
ylabel('amp')
grid on
title('negaitive trig')
title( sprintf('positive trig, subject %i, ch %i', subject, ch))


%% --- classifier from Izabela
classes=[];
features=[];
for file=1:size(dat,2)
    classes=[classes; dat{1,file}.trig];
    for trial=1:size(dat{1,file}.dat,2)
        features=[features reshape(cell2mat(dat{1,file}.dat(1,trial)),[],1)];
    end
end
save('classes_features.mat','classes','features');

% %%
classifier=2; %1-LDA, 2-SVM

testsNo=10;
indexes=crossvalind('Kfold', length(classes), testsNo);

trenAcc=[]; testAcc=[];
for testNo=1:testsNo
    testInd=find(indexes==testNo);
    trenInd=setxor(1:length(classes),testInd);
    
    trenFeatures=features(:,trenInd);
    trenClasses=classes(trenInd);
    
    testFeatures=features(:,testInd);
    testClasses=classes(testInd);

    if classifier==1    
        classifierModel=fitcdiscr(trenFeatures',trenClasses);
    else
        classifierModel=fitcsvm(trenFeatures',trenClasses);
    end
    classifierTrainClasses = predict(classifierModel,trenFeatures');
    results=confusionmat(classifierTrainClasses,trenClasses);
    trenAcc(testNo)=trace(results)/sum(sum(results));

    classifierTestClasses = predict(classifierModel,testFeatures');
    results1=confusionmat(classifierTestClasses,testClasses);
    testAcc(testNo)=trace(results1)/sum(sum(results1));
    
end

trenAccMean=mean(trenAcc)
testAccMean=mean(testAcc)



%% ------- old fashion methods ------------------------------------
%% extract all data from dat
% allData = cell;
for i = 1:5
    allData{i,1} = dat{1,i}.trig;
    temp = reshape( dat{1,i}.dat, 1, 1, numel( dat{1,i}.trig)); 
    allData{i,2} = cell2mat( temp);
end
allTrig = cell2mat(allData(:,1));
allData = cell2mat( reshape( allData(:,2),1,1,5));

%% try to apply spikes sorting method (jrclust). Use data from all subjects
% Ref: doi: https://doi.org/10.1101/101030  (James J. Jun et al 2017)

% find the local maximum
tRange = [.25 .4]*250;
tRange = [ floor(tRange(1)) : ceil(tRange(2)) ];

tShift = 10; % 1 point = 4 ms (fs=250Hz)

% get the channel contains local max between p200-p400
[~,ch] = max( abs( allData( tRange,:,:)));
[~,ch] = max( ch);
% get mean waveform from all postitive/negative trials
allposMean = mean( allData( tRange,:,allTrig==1), 3);
allnegMean = mean( allData( tRange,:,allTrig==-1), 3);

% %% calculate covariance between mean waveform and all positive trigs
cov1_pos = zeros( sum(allTrig==1), 1);
ind = find( allTrig==1);
for i = 1:numel(cov1_pos)
    temp = cov( allData(tRange,:,ind(i)), allposMean); 
    cov1_pos(i) = temp(1,2);
end
cov2_pos = zeros( size( cov1_pos));

allposMean_shift = circshift( allposMean, tShift, 2);
for i = 1:numel(cov2_pos)
    temp = cov( allData(tRange,:,ind(i)), allposMean_shift); 
    cov2_pos(i) = temp(1,2);
end

% calculate covariance between mean waveform and all negative trigs
cov1_neg = zeros( sum(allTrig==-1), 1);
ind = find( allTrig==-1);
for i = 1:numel(cov1_neg)
    temp = cov( allData(tRange,:,ind(i)), allnegMean); 
    cov1_neg(i) = temp(1,2);
end
cov2_neg = zeros( size( cov1_neg));

allnegMean_shift = circshift( allnegMean, tShift, 2);
for i = 1:numel(cov2_neg)
    temp = cov( allData(tRange,:,ind(i)), allnegMean_shift); 
    cov2_neg(i) = temp(1,2);
end

% %% check by plotting the cov
figure; 
hold on
plot( cov1_pos, cov2_pos, 'ko', 'DisplayName', 'pos trig');
plot( cov1_neg, cov2_neg, 'ro', 'DisplayName', 'neg trig');
title( sprintf('Covariance clustering. Data from all subjects. Time shift = %i ms', tShift*4));
xlabel('covariance 1')
ylabel('covariance 2')
grid on
set(gca, 'FontSize', 14);
%% try to apply spikes sorting method (jrclust). Check data from each subject.

subject = 5;
tShift = 5; % 1 point = 4 ms (fs=250Hz)

figure;
for s = 1:5
    range = [1:1200]+1200*(s-1);
    partData = allData(:,:,range);
    partTrig =  allTrig(range);
    
    % %% calculate find the local maximum
    tRange = [.25 .4]*250;
    tRange = [ floor(tRange(1)) : ceil(tRange(2)) ];
    
    % get the channel contains local max between p200-p400
    [~,ch] = max( abs( partData( tRange,:,:)));
    [~,ch] = max( ch);
    % get mean waveform from all postitive/negative trials
    allposMean = mean( partData( tRange,:,partTrig==1), 3);
    allnegMean = mean( partData( tRange,:,partTrig==-1), 3);
    
    % get covariance from mean waveform from positive trig
    cov1_pos = zeros( sum(partTrig==1), 1);
    ind = find( partTrig==1);
    for i = 1:numel(cov1_pos)
        temp = cov( partData(tRange,:,ind(i)), allposMean);
        cov1_pos(i) = temp(1,2);
    end
    cov2_pos = zeros( size( cov1_pos));
    
    allposMean_shift = circshift( allposMean, tShift, 2);
    for i = 1:numel(cov2_pos)
        temp = cov( partData(tRange,:,ind(i)), allposMean_shift);
        cov2_pos(i) = temp(1,2);
    end
    
    % %%
    % get covariance from mean waveform from negative trig
    cov1_neg = zeros( sum(partTrig==-1), 1);
    ind = find( partTrig==-1);
    for i = 1:numel(cov1_neg)
        temp = cov( partData(tRange,:,ind(i)), allnegMean);
        cov1_neg(i) = temp(1,2);
    end
    cov2_neg = zeros( size( cov1_neg));
    % tShift = 5; % 1 point = 4 ms (fs=250Hz)
    allnegMean_shift = circshift( allnegMean, tShift, 2);
    for i = 1:numel(cov2_neg)
        temp = cov( partData(tRange,:,ind(i)), allnegMean_shift);
        cov2_neg(i) = temp(1,2);
    end
    
    % check by plotting the cov
    subplot(1,5,s)
    hold on
    plot( cov1_pos, cov2_pos, 'ko', 'DisplayName', 'pos trig');
    plot( cov1_neg, cov2_neg, 'ro', 'DisplayName', 'neg trig');
    title( sprintf('subject %i, time shift = %i ms', s, tShift*4 ));
    
end









