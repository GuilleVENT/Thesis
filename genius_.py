# imports

import os
import requests
from pathlib import Path
import pandas as pd

from termcolor import colored
from bs4 import BeautifulSoup as bs
import json as js

import re

import urllib

import sys


##########################################
############## access token ##############

clientID = "nirnV08m0NeV9MArdqmw01KngDBmnbfgfTdZTGJjtT0cV-ePMJTr2KFOIzEqvjAp"
clientSecret = "MXT-mfboPvtY3K78B0r225-PyuiD5Ug9r5VzWtuuYXApECZ_-wJi73vsf28hC-7Cz-JpmE5BM4i2dBBR4zW6tw"
token = "NdUWvm-oduXwnXgk5qKhszy6-3S534t93rDwdP2nqA9NynZllsBNBU9ByEUzqjBB"

# TOKEN
parameters = {'client_id':clientID,
			'redirect_uri':"http://localhost:8000/",
			'scope':'',
			'state':'1',
			'response_type':"code"}

genius_auth = "https://api.genius.com/oauth/authorize"


genius_url = "http://api.genius.com"
headers = {'Authorization': 'Bearer '+token}

############## access token ##############
##########################################


# HARDCODED:
########################################
############## SONG INPUT ##############

#song_name 	= "Bohemian Rhapsody"
#artist_name = "Queen"

############## SONG INPUT ##############
########################################


def genius_API(song_api_path):
	#print(song_api_path)
	song_api_url = genius_url+song_api_path
	print(song_api_url)

	response = requests.get(song_api_url,headers=headers)

	json = response.json()

	#print(json)

	html_url = json['response']['song']['path']
	print(colored(genius_url+html_url,'cyan'))
	song_link = "http://wwww.genius.com"+html_url
	print(colored(song_link,'magenta'))


	return song_link



def genius_PARSE(song_link):
	response = requests.get(song_link)

	html = bs(response.text, "html.parser")

	#remove script tags that they put in the middle of the lyrics
	[h.extract() for h in html('script')]


	#at least Genius is nice and has a tag called 'lyrics'!
	lyrics = html.find("div", class_="lyrics").get_text() #updated css where the lyrics are based in HTML

	print("################### THIS IS THE SONG ######################")

	print(lyrics)

	print("################### THIS IS THE SONG ######################")

	return lyrics


def init(song_name, artist_name):

	if '(' in song_name:
		song_title_w_feat = song_name 
		print(colored('song with parenthesis','red'))
		song_name_ = re.sub(r'\([^)]*\)', '', song_name) ## featuring song-> artist name in parenthesis.
		song_name = song_name_

	search_url = genius_url+'/search'
	search_data = {'q': artist_name + song_name}

	response = requests.get(search_url, params=search_data, headers=headers)
	json = response.json()
	
	#print(js.dumps(json, indent=2))
	#sys.exit('json')

	if len(json['response']['hits'])==0:
		print(colored("No Success in Genius-API...",'red'))
		print("SONG NAME   :                "+song_name)
		print("ARTIST NAME :                "+artist_name)
		
		#print(hit['result']['primary_artist']['name'])

		#retryed = True                                  ## !! DEVELOPING THIS IS AN ETERNAL LOOP
		#retry(song_name,artist_name,album_name)

		return #None

	## PATH__
	path__ = "/"+artist_name.replace(" ", "-")+"-"+song_name.replace(" ", "-")+"-lyrics"
	## PATH__ 

	for hit in json['response']['hits']:

		#### scraping debugging area ####
		#print("########## REQUEST RESULTS ##########")

		#print(hit['result'])

		#print("########## REQUEST RESULTS ##########")
		#sys.exit()
		song_title_hit        =     hit['result']['title']
		song_title_w_feat_hit =     hit['result']['title_with_featured']

		print("########## SONG NAME ###########")
		print(colored(song_title_hit,'cyan'))
		#print(song_title_w_feat_hit)
		print('########## ARTIST NAME ##########')
		print(colored(hit['result']['primary_artist']['name'],'blue'))
		print("\n")
		#### scraping debugging area ####
		

		if artist_name.lower() in hit['result']['primary_artist']['name'].lower() and song_name.lower() in hit['result']['title']:
			
			print(colored("Success finding Song in Genius-API",'green'))
			song_info = hit

			song_api_path = song_info['result']['api_path']

			## CALL FUNCTIONS

			song_link = genius_API(song_api_path)

			lyrics 	  = genius_PARSE(song_link)
			
			#print("type"+str(type(lyrics)))
			#print(len(str(lyrics)))
			#print(lyrics)

			#lyrics_analysis(lyrics)

			return(lyrics)


		if path__.lower() in hit['result']['path'].lower():
			print('YES - Path_')

			print(colored("Success finding Song in Genius-API",'green'))
			song_info = hit
			print(hit)

			song_api_path = song_info['result']['api_path']

			## CALL FUNCTIONS

			song_link = genius_API(song_api_path)

			lyrics    = genius_PARSE(song_link)

				

			return(lyrics)
			#print("type"+str(type(lyrics)))
			#print(len(str(lyrics)))
			#print(lyrics)

			#lyrics_analysis(lyrics)

			

		print(hit['result']['full_title'].lower().split('by'))
		
		if song_name.lower() in hit['result']['full_title'].lower():
			print('YES - SONG_NAME')
			song_info = hit

			song_api_path = song_info['result']['api_path']

			## CALL FUNCTIONS

			song_link = genius_API(song_api_path)

			lyrics    = genius_PARSE(song_link)
				
			#print("type"+str(type(lyrics)))
			#print(len(str(lyrics)))
			#print(lyrics)

			#lyrics_analysis(lyrics)

			return(lyrics)
		
def main(song_name,artist_name):
	print("SONG:   "+ colored(song_name,'magenta'))
	print("ARTIST: "+ colored(artist_name,'magenta'))

	lyrics = init(song_name, artist_name)
	#print(lyrics)

	## read
	no_lyrics_file = Path(r'lyrics/no_lyrics.tsv')
	headers = ['SONG','ARTIST']

	if not no_lyrics_file.exists():
		no_lyrics_df = pd.DataFrame(columns = headers)
	else:
		# read
		no_lyrics_df = pd.read_csv(no_lyrics_file,sep='\t')


	try:

		directory = r'lyrics/'+artist_name+'/'
		if not os.path.exists(directory):
			os.makedirs(directory)

		'''
		if lyrics == 'error':
			print("NOT DEVELOPED YET")
		'''

		if "/" in song_name:
			song_name = song_name.replace('/','-')
		if "/" in artist_name:
			artist_name = artist_name.replace('/'," ")

		file = open(directory+song_name+'.txt','w') 
		file.write(lyrics) 
		file.close()
	
	except TypeError:
		print(' - Could not find lyrics to... ')
		print("SONG:   "+ colored(song_name,'red'))
		print("ARTIST: "+ colored(artist_name,'red'))
		
		df = pd.DataFrame([[song_name,artist_name]],index=None,columns=headers)
		
		result_df = pd.concat([no_lyrics_df,df],axis=0).drop_duplicates().reset_index(drop=True)
		print(result_df)	

		result_df.to_csv(no_lyrics_file,sep='\t',index=False)
		

#main()