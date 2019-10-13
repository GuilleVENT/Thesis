import sys
import os
import pandas as pd
import urllib
from termcolor import colored
import numpy as np
import requests

import access

def init():
	### READ DATA 
	songs_data = r'songs_data/all_songs.csv'
	df = pd.read_csv(songs_data)

	### GET DATA 
	download_link = df['Download_Link'].tolist()
	artists_list = df['Artist_Name'].tolist()
	songs_ids = df['Song_ID'].tolist()
	#print(download_link)

	for indx, link in enumerate(download_link):
		if link != 'not available':
			try:
				req_spotify_genre(artists_list[indx])
			except:
				pass




def req_spotify_genre(artist_name):
	if ' ' in artist_name:
		artist_name.replace(' ','%20')
	url = "https://api.spotify.com/v1/search?q="+artist_name+"*&type=artist"
	
	headers ={
    	"Authorization": "Bearer "+access_token,
    	}

	response = requests.request("GET", url, headers=headers)
	#print(response)
	#print(response.text)
	#print(response.json())

	data = response.json()
	#print(data)
	#print(data['artists']['items'])
	print(data['artists']['items'][0]['genres'])
	#sys.exit()
	#print(data['audio_features'])
	#list_of_dics = data['audio_features']

	#return list_of_dics

access_token = access.get_token()
init()