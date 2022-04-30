%% Load all data, segmenting into each trial

dat1 = load("./Data/S1.mat");
dat2 =load("./Data/S2.mat");
dat3 =load("./Data/S3.mat");
dat4 =load("./Data/S4.mat");
dat5 =load("./Data/S5.mat");

dat = {dat1,dat2,dat3,dat4,dat5};
findat = {};
fintar = {};
for indexs = 1:numel(dat)
    gbp = dat{indexs};
    A = gbp.trig;
    A1 = gbp.y;

    b = [1, -1];
    c = ismember(A, b);
    % Extract the elements of a at those indexes.
    indexes = find(c);

    start_index  = [indexes; size(A, 1) + 1];
    end_index = start_index + 125; % 0-500ms
    n  = numel(start_index) - 1;
    Result = cell(1, n);
    for k = 1:n
        Result{k} = A1(start_index(k):end_index(k)-1, :); 
    end

    % adding segmented trials to data
    gbp.dat = Result;
    gbp.trig = nonzeros(gbp.trig);
    gbp.y = [];
    dat{indexs} = gbp; 

end

save("data.mat",'dat');

