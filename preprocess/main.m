warning off;
listFile= { 'MESMAR', 'ARCALE', 'SCAEMI', 'CILRAM', 'GNATN', 'CULLEO', 'BORGIA', 'VALNIC', 'DESTER', 'SALSTE',
                               'MARFRA', 'ANZALE', 'DIFANT', 'VALPAO', 'BORFRA'};
pathRoot='/tmp/transfer/originEEGData';
num_of_subjects = size(listFile, 2);
pathSaveRoot='/tmp/transfer';

FS=200; % fs是采样率，我们的数据用的是200hz
%SampleTime=1;  % time是一个样例持续的时间，这里我们取1s
%SampleNum=400; % samplenum是正例和反例的数目，这里取700，即最后获得700正例+700反例
channelNum=61; % 通道数目，本例中用的61

% 获得不同样本持续时间、不同重叠比例、不同数目的多个特征提取结果
SampleTimeList=[0.8,0.9,0.8,0.9];
chongDieList=[0,0,0.1,0.1];
chongDieNumList=[450,400,530,450];
for index_of_chongdieIndex=1:size(chongDieList, 2)
    chongDie=chongDieList(1,index_of_chongdieIndex);
    SampleNum=chongDieNumList(1,index_of_chongdieIndex);
    SampleTime=SampleTimeList(1,index_of_chongdieIndex);
    pathSaveSaveRoot=[pathSaveRoot '/' num2str(SampleTime) '_' num2str(chongDie) '_' num2str(SampleNum) '/'];
    if ~exist(pathSaveSaveRoot,'dir')
      mkdir(pathSaveSaveRoot);
    end
    for index_of_subject = 1:num_of_subjects
        this_subject = listFile{index_of_subject};
        fprintf(1, ['\n=================' index_of_subject ':' this_subject '\t start====================\n']);
        [Sdata,label,Smrk]=get_data([pathRoot '\' this_subject],FS,SampleNum,channelNum,SampleTime,chongDie);
        data = featext(Sdata,Smrk,FS,SampleTime);
        save([pathSaveSaveRoot  this_subject '.mat'], 'data','label');
        fprintf(1, ['\n=================' index_of_subject ':' this_subject '\t close====================\n']);
        clear data label Sdata label Smrk;
    %      load([pathRoot '\' this_subject '\TAV3.mat']);
    %      fprintf(1,[this_subject ' : %d %d\n'], size(data,1),size(data,2) );
    end
end

for index_of_subject = 1:num_of_subjects
    this_subject = listFile{index_of_subject};
    load([pathRoot '\' this_subject '\TAV3.mat']);
    fprintf(1,[this_subject ' : %d %d\n'], size(data,1),size(data,2) );
end
