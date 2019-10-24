file = 'TEMP.mp3'

disp(file)
% This function will calculate all features of a given song 
tic 
[y, sr] = audioread(file);
disp('Time to load song')
toc

[m,n] = size(y)        % dimensions
song_size = [m,n];



%% STEREO vs MONO (we want the audio in Mono) 
% stereo:
if n == 2
    y_left  = y(:,2);
    y_right = y(:,1);

    y_mono  = (y_left + y_right)/2;
    %y_mono2 = (y_left + y_right)/2;
    disp('Song converted to mono')
    
% mono:
else
    y_mono = y;
    disp('Song was already mono')

end

%% Only choose 30s of the Audio. This is the preset when downloading the Spotify Tracks, but a local (genre) library should be cut to 30s for fairness

% song longer than 30s:
if m > 1324153
    disp('Song is longer than 30s')
    
    % trim start
    y_mono(1:1323000) = [];
    
    % trim end
    y_mono(end-1323000:end) = [];
    
    % result: song -30s at the beginning and -30s at the end 
    
    [m2,n_mono] = size(y_mono);
    song_size = [m2,n_mono];
    
    upper_limit = m2-1350000;
    clearvars random_int
    random_int = randi([1 upper_limit],1,1);

    y_mono_def = y_mono(random_int : random_int + 1323000);
    [m3,n_mono] = size(y_mono_def);
    song_size = [m3,n_mono];
    
    y_mono = y_mono_def;
end

    disp('here')
    
    %sound(y_mono,sr)

    x       = miraudio(y_mono);

    f  = mirfilterbank(x,'NbChannels',10)



    frames = mirframe(x)

    spectrum = mirspectrum(frames)



    spect_centroid      = mircentroid(spectrum)
    disp(spect_centroid)
    rolloff_85          = mirrolloff(spectrum)
    disp(rolloff_85)
    
    
    
