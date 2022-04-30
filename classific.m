% clear all
% load data_prepro.mat
% 
% classes=[];
% features=[];
% for file=1:size(dat,2)
%     classes=[classes; dat{1,file}.trig];
%     for trial=1:size(dat{1,file}.dat,2)
%         features=[features reshape(cell2mat(dat{1,file}.dat(1,trial)),[],1)];
%     end
% end
% save('classes_features.mat','classes','features');

classifier=1; %1-LDA, 2-SVM

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

