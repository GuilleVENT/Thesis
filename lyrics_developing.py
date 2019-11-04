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

import L_analysis


import sys




PATH = os.path.dirname(os.path.abspath(__file__))+'/'

path_2_allsongs = PATH+'songs_data/all_songs.tsv'

path_2_lyrics = PATH+'lyrics/'



### GENIUS DATA:
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


def init():

	

	path_2_setlist = PATH+'pl_setlist/'
	PL_data = PATH+'PL_DATA/'

	headers = ['Song_ID','Language','...']

	for user in os.listdir(path_2_setlist):
		
		## developing:
		## user hardcoded	
		if user == 'spotify':

			path_ = path_2_setlist+user+'/'
			for pl in os.listdir(path_):
				file = path_+pl
				print(" Path to Playlist Setlist File")
				print(file)
				print(" Playlist:")
				print(colored(pl[:-4],'cyan'))
				print("  by user:")
				print(colored(user,'cyan'))


				## if path not exist
				if not os.path.exists(PL_data+user+'/'+pl[:-4]+'/'):
					os.makedirs(PL_data+user+'/'+pl[:-4]+'/')

				## check if Lyrics Features already exist
				## update
				output_file = PL_data+user+'/'+pl[:-4]+'/Lyrics_features.tsv'
				try:
					features_df = pd.read_csv(output_file,sep='\t')
					print(' Lyrics Features prior')
					print(features_df)
					empty=False
					if len(features_df['Song_ID'])==0:
						empty = True
				except FileNotFoundError:
					### WRITE DataFrame
					print(colored(' File Not Found','red'))
					features_df = pd.DataFrame(columns = headers)
					empty=True
					######

				###### 

				# read playlist-setlist
				setlist = pd.read_csv(file,sep='\t')
				print(setlist)
				
				songs_ids = setlist['Song2'].tolist()
				del songs_ids[0]	## = ID 
				
				songs_list = setlist['Song1'].tolist()
				del songs_list[0]	## = Name

				artists_list = setlist['Artist1'].tolist()
				del artists_list[0]	## = Name


				print(songs_list)
				print(artists_list)


				for indx, song_name in enumerate(songs_list):
					artist_name = artists_list[indx]

					print("SONG:   "+ colored(song_name,'magenta'))
					print("ARTIST: "+ colored(artist_name,'magenta'))

					## tweak names:
					song_name_, artist_name_ = tweak_names(song_name,artist_name)
					song_name = song_name_
					artist_name = artist_name_

					## check if LYRICS-file already exists:
					path_2 = path_2_lyrics+artist_name+'/'+song_name+'.txt'

					if os.path.isfile(path_2) == False: 
						print('- These lyrics were '+colored('not','red')+ ' downloaded before')
						lyrics = call_genius(song_name, artist_name)

						if lyrics != 'error':
							
							print(user)
							print(pl[:-4])
							print(song_name)
							print(artist_name)

							## Save lyrics
							directory = PATH+'lyrics/'+artist_name+'/'
		
							if not os.path.exists(directory):
								os.makedirs(directory)

							file = open(directory+song_name+'.txt','w') 
							file.write(lyrics) 
							file.close()

							## instantly read this file for analysis!
							lyrics_file = PATH+'lyrics/'+artist_name+'/'+song_name+'.txt'

							lyrics_analysis(lyrics_file)
							
							#sys.exit('success') ## start lyrics analysis function. These are the inputs. 
							## prepare Dataframe before! 
							## this is for appending line to lyrics dataframe

						else: 	# lyrics == 'error'
							print('error-passing')
							pass
							## write NaNs in Lyrics DF! 


					else:
						print(colored('- These lyrics were already downloaded','green'))
						lyrics_file = path_2

						lyrics_analysis(lyrics_file)

						print("--->"+str(path_))
						print(user)
						print(pl[:-4])
						print(song_name)
						print(artist_name)

						## check if line in dataframe exists.

						## read file 
						## call analysis 
						


def lyrics_analysis(lyrics_file):

	## have structure separated from lyrics!
	lyrics, structure	 = 	L_analysis.get_lyrics_from_txt(lyrics_file)
	## to-do --> features of structure 
	## count verses in each estrofa --> insides of the song's structure 
	## DO THIS 05.11

	## feature extraction: LANGUAGE   and  lang_mix (= if a song contains more than one language)
	Lang, Lang_mix 		 =  L_analysis.get_language(lyrics)

	## 

	## text preprocessing
	print("LYRICS TOKENIZATION...")
	lyrics_tokens	 	 = 	L_analysis.text_preprocessing(lyrics)

	rep, rep_100 		 =  L_analysis.repeated_words_feature(lyrics_tokens)

	print(lyrics_tokens)
							


	## return data for DF 

def tweak_names(song_name,artist_name):
	A = False

	if '(' in song_name:
		song_title_w_feat = song_name 
		print('song with parenthesis')
		song_name_ = re.sub(r'\([^)]*\)', '', song_name) ## featuring song-> artist name in parenthesis.
		song_name = song_name_
		A = True
		

	if '/' in song_name:
		song_name.replace('/',' ')
		A = True

	if '/' in artist_name:
		artist_name.replace('/',' ')
		A = True

	if '-' in song_name: ## = radio edits // remixes - let's get OG
		
		song_name_ = song_name.split('-',1)
		song_name = song_name_[0]
		A = True

	if song_name.endswith(' '):
		song_name = song_name
		A = True
	
	if A == True:
		print(' Tweaked song name:')
		print(song_name)

	return song_name, artist_name
					

def call_genius(song_name, artist_name):


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

		return 'error'

	## PATH__
	path__ = "/"+artist_name.replace(" ", "-")+"-"+song_name.replace(" ", "-")+"-lyrics"
	## PATH__ 

	for index, hit in enumerate(json['response']['hits']):

		#### scraping debugging area ####
		#print("########## REQUEST RESULTS ##########")

		#print(hit['result'])

		#print("########## REQUEST RESULTS ##########")
		#sys.exit()
		song_title_hit        =     hit['result']['title']
		song_title_w_feat_hit =     hit['result']['title_with_featured']

		print("########## SONG NAME "+colored(str(index),'magenta')+" ###########")
		print(colored(song_title_hit,'cyan'))
		#print(song_title_w_feat_hit)
		print('########## ARTIST NAME '+colored(str(index),'magenta')+' ##########')
		print(colored(hit['result']['primary_artist']['name'],'blue'))
		print("\n")
		#### scraping debugging area ####
		

		if artist_name.lower() in hit['result']['primary_artist']['name'].lower() and song_name.lower() in hit['result']['title']:
			
			print(colored("Success finding Song in Genius-API",'green'))
			song_info = hit

			song_api_path = song_info['result']['api_path']

			## CALL FUNCTIONS

			lyrics = genius_API(song_api_path)

			return(lyrics)

		else:
			if path__.lower() in hit['result']['path'].lower():
				print('YES - Path_')

				print(colored("Success finding Song in Genius-API",'green'))
				song_info = hit
				print(hit)

				song_api_path = song_info['result']['api_path']

				## CALL FUNCTIONS

				lyrics = genius_API(song_api_path)
		
				return(lyrics)



			print(hit['result']['full_title'].lower().split('by'))
		
			if song_name.lower() in hit['result']['full_title'].lower():
				print('YES - SONG_NAME')
				song_info = hit

				song_api_path = song_info['result']['api_path']

				## CALL FUNCTIONS

				lyrics = genius_API(song_api_path)

				return(lyrics)

			## more ideas?

	print(" ------------   FAIL  ------------- ")
	#print(json['response']['hits'])
	print(js.dumps(json['response']['hits'], indent=4, sort_keys=True))

	## if non of the hits gave results write to tsv file no-lyrics
	# 
	return 'error'


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

	print(" "+colored('LINK','blue')+" to parse")
	print(song_link)

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



init()






