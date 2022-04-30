%% Load all data, segmenting into each trial

dat1 = load("S1.mat");
dat2 =load("S2.mat");
dat3 =load("S3.mat");
dat4 =load("S4.mat");
dat5 =load("S5.mat");

dat = {dat1,dat2,dat3,dat4,dat5};
findat = {};

for indexs = 1:numel(dat)
    gbp = dat{indexs}
    A = gbp.trig;
    A1 = gbp.y;

    b = [1, -1];
    c = ismember(A, b)
    % Extract the elements of a at those indexes.
    indexes = find(c)

    index  = [indexes; size(A, 1) + 1];
    n      = numel(index) - 1;
    Result = cell(1, n);
    for k = 1:n
        Result{k} = A1(index(k):index(k+1)-1, :);  % [EDITED]
    end

    % adding segmented trials to data
    gbp.dat = Result;
    gbp.trig = nonzeros(gbp.trig);
    gbp.y = [];
    dat{indexs} = gbp; 


end

save("data.mat",'dat');