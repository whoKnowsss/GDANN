function X = extractdata(cnt,mrk,nfo,dr)
% Syntax: X = extractdata(cnt, mrk, nfo, dr)
% This function extracts samples from loaded dataset.
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
%   X: dataset as a matrix of order (#cues x #samples_per_cue x #channels)

fprintf(1, '\t\t ... extracting sampes to form dataset\n');
X = zeros(length(mrk), dr*(nfo), size(cnt,2)); % Initialization

for i = 1 : length(mrk)
    X(i, :, :) = cnt(mrk(i) : mrk(i) + dr*(nfo) - 1, :);
end

% Last modified by Monalisa Pal on 07/12/2016.
end