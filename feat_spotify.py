import requests
import json
import access
import pandas as pd
import sys


song_id1="00aNyosmX1zpR0TjJV0Dte"
song_id2="2Ue89xoJHnJUaQBjafP9JH"
songs_list = [song_id1,song_id2]


def init():
	pd.set_option('display.max_columns', None)

	## READ ALL_SONGS
	all_songs=pd.read_csv(r'songs_data/all_songs.csv')
	songs_ids = all_songs['Song_ID'].tolist()


	### WRITE DataFrame			## guille mira a ver si estan todas de nuevo
	headers = ['Song_ID',"danceability","energy","key","loudness","mode","speechiness","acousticness","instrumentalness","liveness","valence","tempo","duration_ms",'time_signature','analysis_url']
	audio_features_df = pd.DataFrame(columns = headers)


	## getting the top 20 songs of the list
	while len(songs_ids) > 20:
		## get 20 songs
		songs_list=songs_ids[:20]
		## del 20 songs from list
		songs_ids = songs_ids[20:]

		features = req_spotify_songs_list(songs_list)

		audio_features_df = order_data(features,audio_features_df)

		audio_features_df.to_csv(r'songs_data/SPOTIFY_features.csv')

		

	if len(songs_ids) <= 20:
		songs_list=songs_ids[:len(songs_ids)]
		## del 20 songs from list
		songs_ids = songs_ids[len(song_ids):]

		features = req_spotify_songs_list(songs_list)

		audio_features_df = order_data(features,audio_features_df)

		audio_features_df.to_csv(r'songs_data/SPOTIFY_features.csv')


def order_data(features,audio_features_df):
	for dictionary in features:
		try:
			dictionary.pop('type')
			dictionary.pop('uri')
			dictionary.pop("track_href")
			dictionary['Song_ID'] = dictionary.pop('id')
			print(dictionary)

			audio_features_df = append_data(dictionary,audio_features_df)
		except AttributeError:
			print(dictionary)
			pass

	return audio_features_df	
	#print(features)
	#df = pd.DataFrame(features)
	#print(df)
	#return df
	#sys.exit()

def append_data(dictionary,audio_features_df):
	audio_features_df = audio_features_df.append(dictionary, ignore_index=True)
	print(audio_features_df)
	return audio_features_df

def req_spotify_songs_list(songs_list):
	
	url = "https://api.spotify.com/v1/audio-features/?ids="+",".join(str(x) for x in songs_list)
	
	headers ={
    	"Authorization": "Bearer "+access_token,
    	}

	response = requests.request("GET", url, headers=headers)
	#print(response)
	#print(response.text)
	#print(response.json())

	data = response.json()
	print(data['audio_features'])
	list_of_dics = data['audio_features']

	return list_of_dics


	


access_token = access.get_token()

init()
#req_spotify_songs_list(songs_list)
