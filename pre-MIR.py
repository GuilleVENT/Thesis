import os
import pandas as pd
from pathlib import Path
import urllib
from termcolor import colored
import numpy as np
import sys

temp_path  = r'songs_data/temp/'
if not os.path.exists(temp_path):
	os.makedirs(temp_path)


## START MATLAB
print("	STARTING MATLAB ENGINE")
import matlab.engine
eng = matlab.engine.start_matlab()
print("		Engine running")


def init():

	PATH = os.path.dirname(os.path.abspath(__file__))+'/'

	path_2_allsongs = PATH+'songs_data/all_songs.tsv'
	file = Path(PATH+'songs_data/songs_MIRfeatures.tsv')

	### WRITE DataFrame
	headers = ['Song_ID','spect_centroid_v' ,'rolloff85_mean', 'rolloff85_std','brightness_mean','brightness_std','spectralflux_mean','spectralflux_std','zerocross_rate','RMS_energy_mean','RMS_energy_std','lowenergy_v','tempo_v','pulseclarity_v','onsets_mean','onsets_std','onsets_peakpos','onsets_peakval','key_v' ,'key_mode_v' , 'ASR_v', 'mfcc_mean' ,'mfcc_std' ,'pitch_mean', 'pitch_std','event_dens_v', 'roughness_mean','roughness_std','cepstrum_mean','cepstrum_std','chroma_mean','chroma_std', 'key_st_mean','key_st_std','key_mode_v','ton_centroid_mean','ton_centroid_std']
	
	## read TO DO !!! 

	audio_features_df = pd.DataFrame(columns = headers)

	if not file.exists():
		audio_features_df.to_csv(file,index=False,sep='\t')

	else:
		#print(file)
		audio_features_df = pd.read_csv(file,sep='\t')
	

	all_songs = pd.read_csv(path_2_allsongs,sep='\t')

	reduced_df = all_songs[['Song2','Preview_URL']]
	print(reduced_df)

	for index, row in reduced_df.iterrows():
		song_id 	= row['Song2']
		song_link	= row['Preview_URL']
		if song_link.startswith('https://'):

			## update
			audio_features_df = pd.read_csv(file,sep='\t')

			## download
			download_30s(song_id,song_link)
			##
			name_mp3 = temp_path+song_id+'.mp3'
			##  MATLAB
			feat_vector = eng.features_ext_3(name_mp3)
			##
			#print(type(feat_vector))
			feat_ = list(np.array(feat_vector._data))
			#print(feat_)
			feat_.insert(0, song_id)
			
			print(' -- Audio Features --')
			print(feat_)
			print(' --     (as DF)    --')

			df = pd.DataFrame([feat_],columns= headers)
			
			print(df)
			print(' --                --')

			result_df = pd.concat([audio_features_df,df])
			print(result_df)

			

			result_df.to_csv(file,sep='\t',index=False)



def download_30s(song_id,song_link):

	## TEMPORARY FILE 
	name_mp3 = temp_path+song_id+'.mp3'
	
	## DOWNLOAD
	print('...downloading')
	urllib.request.urlretrieve(song_link, name_mp3)
	print('Downloaded:')
	print("		"+colored(song_id+'.mp3','green'))










init()