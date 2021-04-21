function [dataall,lable,mark]=get_data(name,fs,sampleNum,channelNum,SampleTime,chongDie)
%  dataall是返回的这个人的eeg数据  length*channelNum (这里用的通道是61)
%  lable是返回的每个样例的标签，0是正常，1是这个人在疲劳驾驶
%  mark是这个样例的起始次序（从1开始）
%  fs是采样率，我们的数据用的是200hz
%  time是一个样例持续的时间，这里我们取2s
%  故一个样例为fs*time=200*3.5=700个数据

%time=SampleTime;
oneti=fs*SampleTime;   %一个样例的数据




% tdata  drow           1
% tdata  nondrow(TAV3)  0

chongDieTi=fs*chongDie;
tDataTemp=zeros(1000000,channelNum);   %dataall的temp数组
tDataIndex=1;                    %dataall的temp数组的index
tLableTemp=zeros(1,10000);         %lable的temp数组
tLableIndex=1;                    %lable的temp数组的index
tMarkTemp=zeros(1,10000);         %lable的temp数组
tMarkIndex=1;                    %lable的temp数组的index
tdrowPath=[ name '/DROW.mat'];
load(tdrowPath); %加载drow文件
tdrowData=data;
if size(data,1)==channelNum
    tdrowData=data';
end
tdrowIndex=1;   %从第400个开始计算
tnonDrowPath=[ name '/TAV3.mat']; 
load(tnonDrowPath); %加载nondrow文件
tnonDrowData=data;
if size(data,1)==channelNum
    tnonDrowData=data';
end
tnonDrowIndex=1;   %从第400个开始计算
tdrowItr=0;
tnonDrowItr=0;
rng('default');
while 1
   if tnonDrowItr>=sampleNum && tdrowItr>=sampleNum
        break
   end
   if (tnonDrowItr>=sampleNum && tdrowItr<sampleNum) || (tnonDrowItr<sampleNum && tdrowItr<sampleNum && rand()<0.5)
        tLableTemp(1,tLableIndex)=1;  % 1代表疲劳,设置lable
        tLableIndex=tLableIndex+1;
        tMarkTemp(1,tMarkIndex)=tDataIndex; %设置mark，及这个样例的起始位置
        tMarkIndex=tMarkIndex+1;
        tDataTemp(tDataIndex:tDataIndex+oneti-1,:)=tdrowData(tdrowIndex:tdrowIndex+oneti-1,:);
        tDataIndex=tDataIndex+oneti-chongDieTi;
        tdrowIndex=tdrowIndex+oneti-chongDieTi;
        tdrowItr=tdrowItr+1;
   else
        tLableTemp(1,tLableIndex)=0;  % 0代表nondrow
        tLableIndex=tLableIndex+1;
        tMarkTemp(1,tMarkIndex)=tDataIndex; %设置mark，及这个样例的起始位置
        tMarkIndex=tMarkIndex+1;
        tDataTemp(tDataIndex:tDataIndex+oneti-1,:)=tnonDrowData(tnonDrowIndex:tnonDrowIndex+oneti-1,:);
        tDataIndex=tDataIndex+oneti-chongDieTi;
        tnonDrowIndex=tnonDrowIndex+oneti-chongDieTi;
        tnonDrowItr=tnonDrowItr+1;
   end
end




dataall=tDataTemp(1:tMarkTemp(1,2*sampleNum)+oneti-1,:);
lable=tLableTemp(:,1:2*sampleNum);
mark=tMarkTemp(:,1:2*sampleNum);
end

