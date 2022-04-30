clear all

% Balancing data
load data_prepro.mat
load data_prepro_resampled

classes=[];
features=[];
noLeft=sum(dat{1,1}.trig==1);
for file=1:size(dat,2)
    indNonTrialsAll=find(dat{1,file}.trig==-1);
    indNonTrialsChosen=randsample(indNonTrialsAll,noLeft,false);
    indTrialsAll=find(dat{1,file}.trig==1);
    indAll=[indNonTrialsChosen; indTrialsAll];
    indAll=sort(indAll);
    classes=[classes; dat{1,file}.trig(indAll)];
    for trial=1:length(indAll)
        features=[features reshape(cell2mat(dat{1,file}.dat(1,indAll(trial))),[],1)];
    end
end
save('classes_features_balanced.mat','classes','features');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classifier=1;           %1-LDA, 2-SVM
featureSelection=1;     %1-yes, 2-no
featurePercentage=0.2;  %percentage of features left for classification

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
    
    %feature selection with Lasso
    if featureSelection==1
        numberOfFeatures=fix(featurePercentage*size(trenFeatures,1));
        [B,Info]=lasso(trenFeatures', trenClasses,'Alpha',1,'DFmax',numberOfFeatures);
        featuresVector=[];
        for i=1:size(trenFeatures,1)
            if sum(B(i,:))~=0
                featuresVector=[featuresVector; i];
            end
        end
        trenFeatures=trenFeatures(featuresVector,:);
        testFeatures=testFeatures(featuresVector,:);
    end
    
    if classifier==1
        classifierModel=fitcdiscr(trenFeatures',trenClasses);
    else
        classifierModel=fitcsvm(trenFeatures',trenClasses);
    end
    
    classifierTrainClasses = predict(classifierModel,trenFeatures');
    confTrain=confusionmat(classifierTrainClasses,trenClasses);
    trenAcc(testNo)=trace(confTrain)/sum(sum(confTrain));
    
    classifierTestClasses = predict(classifierModel,testFeatures');
    confTest=confusionmat(classifierTestClasses,testClasses);
    testAcc(testNo)=trace(confTest)/sum(sum(confTest));
    
end

lastConfusionTrain=confTrain
lastConfusionTest=confTest
accTren_accTest=[mean(trenAcc) mean(testAcc)]

