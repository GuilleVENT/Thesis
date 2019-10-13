import sys
import os
import pandas as pd
import urllib
from termcolor import colored
import numpy as np


### developing 
## the input of this is songs_download link in a csv list
## it should just take into account playlist that we are studying.
## ATTENTION : <<<< import matlab.engine >>>>
## 					inside init here 


def init():
	### READ DATA 
	songs_data = r'songs_data/all_songs.csv'
	df = pd.read_csv(songs_data)

	### GET DATA 
	download_link = df['Download_Link'].tolist()
	songs_ids = df['Song_ID'].tolist()
	#print(download_link)
	#print(songs_ids)

	### WRITE DataFrame
	headers = ['Song_ID','spect_centroid_v' ,'rolloff85_mean', 'rolloff85_std','brightness_mean','brightness_std','spectralflux_mean','spectralflux_std','zerocross_rate','RMS_energy_mean','RMS_energy_std','lowenergy_v','tempo_v','pulseclarity_v','onsets_mean','onsets_std','onsets_peakpos','onsets_peakval','key_v' ,'key_mode_v' , 'ASR_v', 'mfcc_mean' ,'mfcc_std' ,'pitch_mean', 'pitch_std','event_dens_v', 'roughness_mean','roughness_std','cepstrum_mean','cepstrum_std','chroma_mean','chroma_std', 'key_st_mean','key_st_std','key_mode_v','ton_centroid_mean','ton_centroid_std']
	audio_features_df = pd.DataFrame(columns = headers)

	##  CREATE TEMPORARY FOLDER FOR tomparary storage of the mp3 
	temp = r'songs_data/temp/'
	if not os.path.exists(temp):
		os.makedirs(temp)


	## START MATLAB
	print("STARTING MATLAB ENGINE")
	import matlab.engine
	eng = matlab.engine.start_matlab()
	print("		Engine running")



	## DOWNLOAD mp3
	not_available_count = 0
	yes_available_count = 0 
	for indx, link in enumerate(download_link):
		if not link.startswith('https://'):
			if link == 'not available':
				not_available_count += 1 
				print("Song not available to download:	"+colored(songs_ids[indx],'red'))
			else:
				sys.exit('ERROR: NOT not available, nor link found ')
		else:
			
			#print(audio_features_df)
			
			## TEMPORARY FILE 
			name_mp3 = temp+songs_ids[indx]+'.mp3'
			
			## DOWNLOAD
			print('...downloading')
			urllib.request.urlretrieve(link, name_mp3)
			print('Downloaded:')
			print("		"+colored(songs_ids[indx]+'.mp3','green'))
			
			## MATLAB
			feat_vector=features(eng,name_mp3) ## missing return
			row = [songs_ids[indx]]
			for value in feat_vector:
				row.append(value)
			

			## DF
			print('NEW ROW:')
			print(row)
			#new_row = pd.DataFrame([row], columns=headers)
			yes_available_count += 1
			
			## in case this is parallelized, so we don't overwrite rows 
			#if os.path.isfile(r'songs_data/MIR_audio_features.csv'):


			audio_features_df.loc[yes_available_count] = row
			#print(audio_features_df)

			## SAVE FILE 
			audio_features_df.to_csv(r'songs_data/MIR_audio_features.csv')

			## DEL TEMPORARY FILE
			os.remove(name_mp3)



	print("Songs available to download:")
	print("		"+colored(yes_available_count,'green'))
	print("Songs not available to download:")
	print("		"+ colored(not_available_count,'red'))


#feat_names = 
#feat_names = 
#feat_names = [spect_centroid_v ,rolloff85_mean, rolloff85_std,brightness_mean,brightness_std,spectralflux_mean,spectralflux_std,zerocross_rate,RMS_energy_mean,RMS_energy_std,lowenergy_v,tempo_v,pulseclarity_v,onsets_mean,onsets_std,onsets_peakpos,onsets_peakval,key_v ,key_mode_v , ASR_v, mfcc_mean , mfcc_std ,pitch_mean, pitch_std,event_dens_v, roughness_mean,roughness_std, cepstrum_mean,cepstrum_std,chroma_mean,chroma_std, key_st_mean,key_st_std,key_mode_v,ton_centroid_mean,ton_centroid_std]

def features(eng,name_mp3):
	a=eng.features_ext_3(name_mp3)
	#print(a)
	#print(type(a))
	#print(type(a[0]))
	#print(type(a[0][0]))
	arr=np.asarray(a[0])
	print(arr)
	#print(type(arr))

	return arr.tolist()
	




init()



sys.exit('developing')


'''
I need to update matlab now that i am running python 3.7 , my version of matlab is 2017 which
python engine doesn't support 3.7 only 2.7 , 3.4 , 3.5. 
I need to update Matlab and correctly choose the libraries that I need for MIR Toolbox
--> done 
'''



#eng.features_s_4_TUM(nargout=0)




'''
MULTIPLE ENGINES
eng1 = matlab.engine.start_matlab()
eng2 = matlab.engine.start_matlab()


'''

sys.exit()


## SONG MOOD
import get_mood_HPC

'''
sys.path.append('/nas/ei/home/ga59qek/spotify')


## LEARN AND CLASSIFIE GENRES
import learn_genres_HPC

## SUMMING MOOD AND GERNES
import music_features_HPC

## RAW AUDIO FEATURES PER PL
import pl_features_HPC
'''