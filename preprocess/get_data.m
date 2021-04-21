function [dataall,lable,mark]=get_data(name,fs,sampleNum,channelNum,SampleTime,chongDie)
%  dataall�Ƿ��ص�����˵�eeg����  length*channelNum (�����õ�ͨ����61)
%  lable�Ƿ��ص�ÿ�������ı�ǩ��0��������1���������ƣ�ͼ�ʻ
%  mark�������������ʼ���򣨴�1��ʼ��
%  fs�ǲ����ʣ����ǵ������õ���200hz
%  time��һ������������ʱ�䣬��������ȡ2s
%  ��һ������Ϊfs*time=200*3.5=700������

%time=SampleTime;
oneti=fs*SampleTime;   %һ������������




% tdata  drow           1
% tdata  nondrow(TAV3)  0

chongDieTi=fs*chongDie;
tDataTemp=zeros(1000000,channelNum);   %dataall��temp����
tDataIndex=1;                    %dataall��temp�����index
tLableTemp=zeros(1,10000);         %lable��temp����
tLableIndex=1;                    %lable��temp�����index
tMarkTemp=zeros(1,10000);         %lable��temp����
tMarkIndex=1;                    %lable��temp�����index
tdrowPath=[ name '/DROW.mat'];
load(tdrowPath); %����drow�ļ�
tdrowData=data;
if size(data,1)==channelNum
    tdrowData=data';
end
tdrowIndex=1;   %�ӵ�400����ʼ����
tnonDrowPath=[ name '/TAV3.mat']; 
load(tnonDrowPath); %����nondrow�ļ�
tnonDrowData=data;
if size(data,1)==channelNum
    tnonDrowData=data';
end
tnonDrowIndex=1;   %�ӵ�400����ʼ����
tdrowItr=0;
tnonDrowItr=0;
rng('default');
while 1
   if tnonDrowItr>=sampleNum && tdrowItr>=sampleNum
        break
   end
   if (tnonDrowItr>=sampleNum && tdrowItr<sampleNum) || (tnonDrowItr<sampleNum && tdrowItr<sampleNum && rand()<0.5)
        tLableTemp(1,tLableIndex)=1;  % 1����ƣ��,����lable
        tLableIndex=tLableIndex+1;
        tMarkTemp(1,tMarkIndex)=tDataIndex; %����mark���������������ʼλ��
        tMarkIndex=tMarkIndex+1;
        tDataTemp(tDataIndex:tDataIndex+oneti-1,:)=tdrowData(tdrowIndex:tdrowIndex+oneti-1,:);
        tDataIndex=tDataIndex+oneti-chongDieTi;
        tdrowIndex=tdrowIndex+oneti-chongDieTi;
        tdrowItr=tdrowItr+1;
   else
        tLableTemp(1,tLableIndex)=0;  % 0����nondrow
        tLableIndex=tLableIndex+1;
        tMarkTemp(1,tMarkIndex)=tDataIndex; %����mark���������������ʼλ��
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

