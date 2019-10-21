import os
import requests
import json
import access
import pandas as pd
import sys
from termcolor import colored


def init2():

	PATH = os.path.dirname(os.path.abspath(__file__))+'/'

	path_2_setlist = PATH+'pl_setlist/'
	headers = ['Song_ID','danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration_ms','time_signature']

	PL_data = PATH+'PL_DATA/'

	if not os.path.exists(PL_data):
		os.makedirs(PL_data)

	for user in os.listdir(path_2_setlist):
		path_ = path_2_setlist+user+'/'
		for pl in os.listdir(path_):
			file = path_+pl
			print(" Path to Playlist Setlist File")
			print(file)
			print(" Playlist:")
			print(colored(pl[:-4],'cyan'))
			print("  by user:")
			print(colored(user,'cyan'))
			

			###### 
			# if path not exist
			if not os.path.exists(PL_data+user+'/'+pl[:-4]+'/'):
				os.makedirs(PL_data+user+'/'+pl[:-4]+'/')

			# read playlist-setlist
			setlist = pd.read_csv(file,sep='\t')
			songs_list = setlist['Song2'].tolist()
			del songs_list[0]	## = ID 		

			if len(songs_list) != 0:

				for song_id in songs_list:

					## check if Spotify Features already exist
					## update
					output_file = PL_data+user+'/'+pl[:-4]+'/Spotify_features.tsv'
					try:
						features_df = pd.read_csv(output_file,sep='\t')
						print(' Spotify Features prior')
						print(features_df)
						empty=False
						if len(features_df['Song_ID'])==0:  ## cannot happen reason: -12 lines
							empty = True
					except FileNotFoundError:
						### WRITE DataFrame
						print(colored(' File Not Found','red'))
						features_df = pd.DataFrame(columns = headers)
						empty=True
						######

					if song_id in features_df['Song_ID'].tolist():
						print(colored(song_id+' \n already in '+ output_file,'green'))
						pass
					

					else:
						print(song_id)
						features_res = req_spotify_song_features(song_id)

						print(features_res)
						if empty== True:
							features_df = features_res
						else:
							features_df = pd.concat([features_df,features_res],axis=0).drop_duplicates().reset_index(drop=True)
						features_df.to_csv(output_file,sep='\t',index=False)
			else:
				print('...')#think of something--> these pl cannot be analyzed




def req_spotify_song_features(song_id):
	
	url = "https://api.spotify.com/v1/audio-features/?ids="+song_id
	
	headers ={
    	"Authorization": "Bearer "+access_token,
    	}

	response = requests.request("GET", url, headers=headers)
	print(response)
	#print(response.text)
	#print(response.json())

	data = response.json()
	print(data)
	#print(data['audio_features'])
	#sys.exit()
	#list_of_dics = data['audio_features']
	if data['audio_features']:
		## 
		## this is yet to be fixed
		##
		sys.exit('crash') 

	dict_ = data['audio_features']

	features_df = order_data(dict_)
	print(features_df)
	

	
	return features_df


def order_data(dict_):																																					## change this vvv 
	print("... PRIOR:")																																					## VV POS: 1 VVVVV 
	headers = ['danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration_ms','time_signature','Song_ID']
	print(dict_)
	#features_df = pd.DataFrame.from_dict(dict_)
	
	##change key = 'id' to 'Song_ID' + make it the first element
	del dict_[0]['type']
	del dict_[0]['uri']
	del dict_[0]['analysis_url']
	del dict_[0]['track_href']
	song_id = dict_[0].get('id')
	del dict_[0]['id']

	d = dict_[0]
	d['Song_ID'] =   song_id
	print(d)

	df = pd.DataFrame.from_dict([d])
	
	res = df

	#print("... A POSTERIORI:")
	#print(res)


	return res



access_token = access.get_token()
init2()