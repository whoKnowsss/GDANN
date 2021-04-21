function X = featext(cnt,mrk,nfo,dr)
% Syntax: X = featext(cnt, mrk, nfo, dr)
% This function takes the pre-processed EEG signals and creates a matrix 
% representing different samples using extractdata() function. Using this 
% matrix, Welch's power spectral density estimates are evaluated which 
% represents the feature vector of each EEG trial sample.
% Input -
%   cnt: the continuous EEG signals, size [time x channels]. 
%   mrk: structure of cue information with fields
%        pos: vector of positions of the cue in the EEG signals given in 
%             unit sample, length #cues
%        y: vector of classes (1, 2, or NaN), length #cues
%        className: cell array of class names.
%   nfo: structure providing additional information with fields
%        name: name of the data set,
%        fs: sampling rate,
%        clab: cell array of channel labels,
%        xpos: x-position of electrodes in a 2d-projection,
%        ypos: y-position of electrodes in a 2d-projection.
%   dr: duration of each cue
% Output - 
%   X: dataset as a matrix of order (#cues x (14 x nchan))

DS = extractdata(cnt,mrk,nfo,dr); % extracting samples to form dataset
% estimating power spectral density on dataset for feature extraction
fprintf(1, '\t\t ... applying PSD to dataset\n');
[nobs, ~, nchan] = size(DS);
pC = zeros(nobs, (nfo / 2) + 1, nchan); % Initialization
for i = 1 : nobs
    for j = 1 : nchan
        % Using Welch's periodogram with 50% overlap and hamming window of 
        % size (sampling_frequency)/2.
        [pC(i,:,j),freqC]=pwelch(DS(i,:,j),hamming(nfo/2),0.5,nfo,nfo);
        fprintf(1,'第 %d 次PSD循环结束！\n', (i-1)*nchan+j); 
    end
end
in = zeros((nfo / 2) + 1, 1);
for i = 1 : (nfo / 2) + 1 % choosing the central beta & mu band for EEG
   % if (freqC(i) >= 4 && freqC(i) <= 7) || (freqC(i) >= 8 && freqC(i) <= 13 || (freqC(i) >= 16 && freqC(i) <= 30)   %对应的频段选择
     if (freqC(i) >= 1 && freqC(i) <= 30)  %对应的频段选择
       in(i) = 1;
    end
end

pC = pC(:, logical(in), :);
X = [];
for i = 1 : nchan
    X = horzcat(X,pC(:,:,i)); % concatenating PSD channel-wise
end
% Last modified by Monalisa Pal on 07/12/2016.
end