# imports
import os
import requests
from pathlib import Path
import pandas as pd

from termcolor import colored
from bs4 import BeautifulSoup as bs
import json as js
import re
import itertools
import urllib
import difflib

import L_analysis

import genius_credentials

import sys


def init():
	PATH = os.path.dirname(os.path.abspath(__file__))+'/'

	path_2_setlist = PATH+'pl_setlist/'

	PL_data = PATH+'PL_DATA/'

	path_2_lyrics = PATH+'lyrics/'

	headers = ['Song_ID','Duet','Hiphop_words_100','Metal_words_100','Pop_words_100','happy_words_100','StopWord_100','Total_Words','Lang_MIX','Language','Repeated_Verses','Repetition_100','avg_verses_per_estrofa','avrg_verse_length','longest_verse_len','max_verses_per_estrofa','min_verses_per_estrofa','shortest_verse_len','total_num_estrofas','E01','E02','E03','E04','E05','E06','E07','E08','E09','E10','E11','E12','E13','E14','E15','P01','P02','P03','P04','P05','P06','P07','P08','P09','P10','P11','P12','P13','P14','P15']

	for user in os.listdir(path_2_setlist):
		
		## developing:
		## user hardcoded	
		#  if user == 'spotify':

			path_ = path_2_setlist+user+'/'
			for pl in os.listdir(path_):
				print(' -- * -- * -- * -- * -- * -- * -- * -- * -- * -- * -- * -- * -- ')

				setlist_file = path_+pl
				print(" Path to Playlist Setlist File")
				print(setlist_file)
				print(" Playlist:")
				print(colored(pl[:-4],'cyan'))
				print("  by user:")
				print(colored(user,'cyan'))


				## if path not exist
				if not os.path.exists(PL_data+user+'/'+pl[:-4]+'/'):
					os.makedirs(PL_data+user+'/'+pl[:-4]+'/')

				## check if Lyrics Features already exist
				## 
				output_file = PL_data+user+'/'+pl[:-4]+'/Lyrics_features.tsv'
				try:
					lyrics_features_df = pd.read_csv(output_file,sep='\t')
					#print(' Audio Features prior')
					#print(audio_features_df)
					
				except FileNotFoundError:
					### WRITE DataFrame
					print(colored(' File Not Found','red'))
					lyrics_features_df = pd.DataFrame(columns = headers)
					lyrics_features_df.to_csv(output_file,sep='\t')
				######

				# read playlist-setlist
				setlist = pd.read_csv(setlist_file,sep='\t')
				#print(setlist)
				
				SETLISTsongs_ids = setlist['Song2'].tolist()
				del SETLISTsongs_ids[0]	## = ID 
				
				SETLISTsongs_list = setlist['Song1'].tolist()
				del SETLISTsongs_list[0]	## = Name

				SETLISTartists_list = setlist['Artist1'].tolist()
				del SETLISTartists_list[0]	## = Name

				#print(SETLISTsongs_ids)
				#print(SETLISTsongs_list)
				#print(SETLISTartists_list)

				for indx, song_name in enumerate(SETLISTsongs_list):
					try:
						lyrics_features_df = pd.read_csv(output_file,sep='\t')
						#print(' Audio Features prior')
						#print(audio_features_df)
					
					except FileNotFoundError:
						### WRITE DataFrame
						print(colored(' File Not Found','red'))
						lyrics_features_df = pd.DataFrame(columns = headers)
						lyrics_features_df.to_csv(output_file,sep='\t')


					artist_name = SETLISTartists_list[indx]
					song_id = SETLISTsongs_ids[indx]

					if song_id not in lyrics_features_df['Song_ID'].tolist():

						print(' '+ song_name +" not in "+ output_file)
						
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

								song_lyricsDF = lyrics_analysis(song_id, lyrics_file, headers)

								append_row_to_DF(song_lyricsDF,output_file)


							else: 	# lyrics == 'error'
								print('error - writting NaNs')
								
								song_lyricsDF = pd.DataFrame([[song_id,float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN')]],columns=headers)
								## writing NaNs in Lyrics DF! 							duet		,'E01',		,'E02'			,'E03'	,'E04'			,'E05		','E06'		,'E07'		,'E08'			,'E09'		,'E10'		,'E11',			'E12',		'E13',		'E14'		,'E15',		'Hiphop_words_100','Lang_MIX','Language','Metal_words_100','P01','P02',			'P03',		'P04',			'P05',		'P06',		'P07',		'P08',		'P09','			P10','		P11',		'P12',		'P13',			'P14',		'P15',		'Pop_words_100','Repeated Verses','Repetition_100','StopWord_100','Total_Words','avg_verses_per_estrofa','avrg_verse_length','happy_words_100','longest_verse_len','max_verses_per_estrofa','min_verses_per_estrofa','shortest_verse_len','total_num_estrofas']
								append_row_to_DF(song_lyricsDF,output_file)
								#


						else:
							print(colored('- These lyrics were already downloaded','green'))
							lyrics_file = path_2

							print(" - User:")
							print(user)
							print(" - Playlist ID:")
							print(pl[:-4])
							
							#print(song_name)
							#print(artist_name)

							song_lyricsDF = lyrics_analysis(song_id,lyrics_file,headers)

							append_row_to_DF(song_lyricsDF,output_file)


def lyrics_analysis(song_id, lyrics_file, headers):

	## have structure separated from lyrics!
	lyrics, structure	 = 	L_analysis.get_lyrics_from_txt(lyrics_file)

	total_num_estrofas = len(structure)
	## 
	## count verses in each estrofa --> insides of the song's structure 
	parts, duet = L_analysis.structure_features(lyrics, structure)
	
	## 
	## 
	verses_per_estrofa, min_verses_per_estrofa, max_verses_per_estrofa, avg_verses_per_estrofa, total_num_estrofas = L_analysis.get_estrofas(lyrics, structure)
	
	shortest_verse_len, longest_verse_len, avrg_verse_length	=  L_analysis.get_lengths(lyrics)	


	## feature extraction: LANGUAGE   and  lang_mix (= if a song contains more than one language)
	language_encoded,lang_mix,language		=  L_analysis.get_language(lyrics)


	repe_100, Num_repeated_verses   = L_analysis.get_repetitions(lyrics)


	## text preprocessing
	## if language == 'en':
	print(" LYRICS TOKENIZATION...")
	lyrics_tokens	 		 = 	L_analysis.text_preprocessing(lyrics)

	sum_,pop_100,metal_100,happy_100,hiphop_100			  	 =  L_analysis.get_words(lyrics_tokens,language)
		
	stopWords_100 		 = L_analysis.get_stopWords(lyrics_tokens,language)
	

	song_DF = create_song_lyricsDF(song_id,parts, duet,verses_per_estrofa, min_verses_per_estrofa, max_verses_per_estrofa, avg_verses_per_estrofa, shortest_verse_len, longest_verse_len, avrg_verse_length, language_encoded,lang_mix,language, repe_100, Num_repeated_verses,sum_,pop_100,metal_100,happy_100,hiphop_100,stopWords_100, total_num_estrofas, headers)
	return(song_DF)

def create_song_lyricsDF(song_id, parts, duet,verses_per_estrofa, min_verses_per_estrofa, max_verses_per_estrofa, avg_verses_per_estrofa, shortest_verse_len, longest_verse_len, avrg_verse_length, language_encoded,lang_mix,language, repe_100, Num_repeated_verses,sum_,pop_100,metal_100,happy_100,hiphop_100,stopWords_100, total_num_estrofas, headers):
	
	'''
	headers = ['Song_ID','Duet','Hiphop_words_100','Metal_words_100','Pop_words_100','happy_words_100','StopWord_100','Total_Words','Lang_MIX','Language','Repeated_Verses','Repetition_100','avg_verses_per_estrofa','avrg_verse_length','longest_verse_len','max_verses_per_estrofa','min_verses_per_estrofa','shortest_verse_len','total_num_estrofas','E01','E02','E03','E04','E05','E06','E07','E08','E09','E10','E11','E12','E13','E14','E15','P01','P02','P03','P04','P05','P06','P07','P08','P09','P10','P11','P12','P13','P14','P15']
	'''

	indx = len(parts)
	while indx<15:
		parts.append(0)
		indx += 1

	indx = len(verses_per_estrofa)
	while indx<15:
		verses_per_estrofa.append(0)
		indx += 1

	DICT = {headers[0]: [song_id],
			headers[1]: [duet],
			headers[2]: [hiphop_100],
			headers[3]: [metal_100],
			headers[4]: [pop_100],
			headers[5]: [happy_100],
			headers[6]: [stopWords_100],
			headers[7]: [sum_],
			headers[8]: [lang_mix],
			headers[9]: [language_encoded],
			headers[10]: [Num_repeated_verses],
			headers[11]: [repe_100],
			headers[12]: [avg_verses_per_estrofa],
			headers[13]: [avrg_verse_length],
			headers[14]: [longest_verse_len],
			headers[15]: [max_verses_per_estrofa],
			headers[16]: [min_verses_per_estrofa],
			headers[17]: [shortest_verse_len],
			headers[18]: [total_num_estrofas],
			headers[19]: [parts[0]],
			headers[20]: [parts[1]],
			headers[21]: [parts[2]],
			headers[22]: [parts[3]],
			headers[23]: [parts[4]],
			headers[24]: [parts[5]],
			headers[25]: [parts[6]],
			headers[26]: [parts[7]],
			headers[27]: [parts[8]],
			headers[28]: [parts[9]],
			headers[29]: [parts[10]],
			headers[30]: [parts[11]],
			headers[31]: [parts[12]],
			headers[32]: [parts[13]],
			headers[33]: [parts[14]],
			headers[34]: [verses_per_estrofa[0]],
			headers[35]: [verses_per_estrofa[1]],
			headers[36]: [verses_per_estrofa[2]],
			headers[37]: [verses_per_estrofa[3]],
			headers[38]: [verses_per_estrofa[4]],
			headers[39]: [verses_per_estrofa[5]],
			headers[40]: [verses_per_estrofa[6]],
			headers[41]: [verses_per_estrofa[7]],
			headers[42]: [verses_per_estrofa[8]],
			headers[43]: [verses_per_estrofa[9]],
			headers[44]: [verses_per_estrofa[10]],
			headers[45]: [verses_per_estrofa[11]],
			headers[46]: [verses_per_estrofa[12]],
			headers[47]: [verses_per_estrofa[13]],
			headers[48]: [verses_per_estrofa[14]],
	}
	
	print('DICT:')
	print(DICT)
	
	print('HEADERS:')
	print(headers)


	song_DF = pd.DataFrame(DICT)

	print(song_DF)
	
	return(song_DF)



def append_row_to_DF(song_lyricsDF,output_file):
	
	song_lyricsDF.to_csv(output_file, sep='\t', mode='a', index= False, header=False)


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


		if index == 1: 
			P_=difflib.SequenceMatcher(None,song_name,song_title_hit).ratio()
			if P_ > 0.5:
				print(colored("Success finding Song in Genius",'green'))
				print(" Method: "+colored('SequenceMatcher1','magenta'))
				## CALL FUNCTIONS
				try:
					url = hit['result']['url']
					lyrics = genius_scrape(url)

				except: ## in case of errors scraping. Let's keep using the API:
					song_api_path = hit['result']['api_path']
					lyrics = genius_API(song_api_path)

				return(lyrics)

			if '(' in song_title_hit:
				song_title_hit_=song_title_hit.split('(')[0]
				P_=difflib.SequenceMatcher(None,song_name,song_title_hit_).ratio()
				if P_ > 0.5:
					print(colored("Success finding Song in Genius",'green'))
					print(" Method: "+colored('SequenceMatcher2','magenta'))
					## CALL FUNCTIONS
					try:
						url = hit['result']['url']
						lyrics = genius_scrape(url)

					except: ## in case of errors scraping. Let's keep using the API:
						song_api_path = hit['result']['api_path']
						lyrics = genius_API(song_api_path)

					return(lyrics)

		

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