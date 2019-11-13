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

import genius_credentials

import sys




PATH = os.path.dirname(os.path.abspath(__file__))+'/'

path_2_allsongs = PATH+'songs_data/all_songs.tsv'

path_2_lyrics = PATH+'lyrics/'




def init():

	path_2_setlist = PATH+'pl_setlist/'
	PL_data = PATH+'PL_DATA/'

	headers = ['Song_ID','Language','...']

	for user in os.listdir(path_2_setlist):
		
		## developing:
		## user hardcoded	
		#if user == 'topsify':

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
					song_id = songs_ids[indx]

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
							print(" - User:")
							print(user)
							print(" - Playlist ID:")
							print(pl[:-4])

							#print(song_name)
							#print(artist_name)

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

						print(" - User:")
						print(user)
						print(" - Playlist ID:")
						print(pl[:-4])
						
						#print(song_name)
						#print(artist_name)

						lyrics_analysis(lyrics_file)

						## check if line in dataframe exists.

						## read file 
						## call analysis 
						


def lyrics_analysis(lyrics_file):

	## have structure separated from lyrics!
	lyrics, structure	 = 	L_analysis.get_lyrics_from_txt(lyrics_file)
	#print(lyrics)
	## to-do --> features of structure 
	## count verses in each estrofa --> insides of the song's structure 
	L_analysis.structure_features(lyrics, structure)

	## 
	## these are 4 features for the Lyrics DF
	total_num_estrofas, max_verses_per_estrofa, min_verses_per_estrofa, avg_verses_per_estrofa,verses_per_estrofa = L_analysis.get_estrofas(lyrics, structure)
	
	shortest_verse_len, longest_verse_len, avrg_verse_length	=  L_analysis.get_lengths(lyrics)	
	
	## feature extraction: LANGUAGE   and  lang_mix (= if a song contains more than one language)
	Lang, Lang_mix 		 =  L_analysis.get_language(lyrics)

	## text preprocessing
	print(" LYRICS TOKENIZATION...")
	lyrics_tokens	 	 = 	L_analysis.text_preprocessing(lyrics)
	total_num_words = L_analysis.get_words(lyrics_tokens,Lang)



	stopwords_100 		 = L_analysis.get_stopWords(lyrics_tokens,Lang)

	#L_analysis.get_repetitions(lyrics_tokens)
	
							


	## return data for DF 

def tweak_names(song_name,artist_name):
	A = False

	if '/' in song_name:
		print('... / in song name...tweaking.. ')
		song_name_	=song_name.replace('/',' ')
		song_name = song_name_
		A = True

	if '(' in song_name:
		song_title_w_feat = song_name 
		print('... song with parenthesis... tweaking.. ')
		song_name_ = re.sub(r'\([^)]*\)', '', song_name) ## featuring song-> artist name in parenthesis.
		song_name = song_name_
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

	token = genius_credentials.get_token()
	headers = {'Authorization': 'Bearer '+token}	
	genius_url = "http://api.genius.com"
	search_url = genius_url+'/search'


	query = song_name + " " + artist_name
	#print(query.encode('utf-8'))
	search_data = {'q': query.encode('utf-8')} ## mal - corregir

	print(search_data)

	try:
		response = requests.get(search_url, params=search_data, headers=headers)
		print(response.status_code)
	except:
		print(response.status_code)
		call_genius(song_name, artist_name) ## retry...
	

	json = response.json()
	
	## debugging:
	#print(" Search Request json Result: ")
	#print(js.dumps(json, indent=2))
	#sys.exit('json')


	if len(json['response']['hits'])==0:
		print(colored("No Success in Genius...",'red'))
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
			
			print(colored("Success finding Song in Genius",'green'))
			print(" Method: "+colored('Double-Match','magenta'))


			## CALL FUNCTIONS
			try:
				url = hit['result']['url']
				lyrics = genius_scrape(url)

			except: ## in case of errors scraping. Let's keep using the API:
				song_api_path = hit['result']['api_path']
				lyrics = genius_API(song_api_path)

			return(lyrics)

		else:
			if path__.lower() in hit['result']['path'].lower():
				print(colored("Success finding Song in Genius",'green'))
				print(" Method: "+colored('Path_','magenta'))
		

				## CALL FUNCTIONS
				try:
					url = hit['result']['url']
					lyrics = genius_scrape(url)

				except:
					song_api_path = hit['result']['api_path']
					lyrics = genius_API(song_api_path)
		
				return(lyrics)



			#print(hit['result']['full_title'].lower().split('by'))
		
			if song_name.lower() in hit['result']['full_title'].lower():
				print(colored("Success finding Song in Genius",'green'))
				print(" Method: "+colored('Song Name','magenta'))
				## CALL FUNCTIONS
				try:
					url = hit['result']['url']
					lyrics = genius_scrape(url)
				except:
					song_api_path = hit['result']['api_path']
					lyrics = genius_API(song_api_path)

				return(lyrics)

			## more ideas?

	print(" ------------   FAIL  ------------- ")
	#print(json['response']['hits'])
	print(js.dumps(json['response']['hits'], indent=4, sort_keys=True))

	## if non of the hits gave results write to tsv file no-lyrics
	# 
	return 'error'

def genius_scrape(song_link):
	print("	This is based on web-scraping.\n 	URL: ")
	print(colored(song_link,'green'))

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

def genius_API(song_api_path):
	token = genius_credentials.get_token()
	headers = {'Authorization': 'Bearer '+token}
	genius_url = "http://api.genius.com"

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

	lyrics = genius_scrape(song_link)

	return lyrics



init()

#lyrics_analysis('/Users/guillermoventuramartinezAIR/Desktop/FP/lyrics/The Clash/Should I Stay or Should I Go .txt')



## debugging GENIUS:
# HARDCODED:
########################################
############## SONG INPUT ##############

#song_name 	= "Purple Rain"
#artist_name = "Prince"

############## SONG INPUT ##############
########################################

#call_genius(song_name,artist_name)

