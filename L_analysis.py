import os
import pandas as pd
import re
import sys

from termcolor import colored

## language detection
from langdetect import detect
import collections
from itertools import dropwhile


## text processing
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
#nltk.download('stopwords')


PATH = os.path.dirname(os.path.abspath(__file__))+'/'


def get_lyrics_from_txt(file):

	with open(file) as f:
		content = f.readlines()

	lyrics = [x.strip() for x in content] ## each line is an element in a list

	structure = []

	for line in lyrics:
		if line.startswith('['):
			structure.append(line)
			lyrics.remove(line)
		'''
		if len(line)==0:
			lyrics.remove(line)
		'''

	print(" 	Song Structure: 	  	")
	print(structure)

	return lyrics, structure 			### lyrics and *structure* !
	
def structure_features():
	pass


def get_lengths(lyrics_tokens):
	#
	#print(lyrics_tokens)
	# not unnesting (verses == sentence)
	
	## delete empty rows:
	#print('-- -- -- -- -- -- -- --')
	lyrics_tokens = list(filter(None, lyrics_tokens))

	#print(lyrics_tokens)

	shortest_verse 		= min(lyrics_tokens, key=len)
	shortest_verse_len 	= len(shortest_verse)

	print(' Shortest Verse:')
	print(shortest_verse)
	print(shortest_verse_len)


	longest_verse  		= max(lyrics_tokens, key=len)
	longest_verse_len 	= len(longest_verse)

	print(' Longest Verse:')
	print(longest_verse)
	print(longest_verse_len)

	verse_length = []
	for verse in lyrics_tokens:
		verse_length.append(len(verse))
	#print(verse_length)
	avrg_verse_length = float(sum(verse_length)/len(verse_length))
	print(' Avrg Verse Length:')
	print(avrg_verse_length)

	return(shortest_verse_len, longest_verse_len, avrg_verse_length)


def get_repetitions():
	## unnesting lyrics:
	lyrics_tokens = sum(lyrics_tokens,[])

	rep = []
	for word in song:
		if word not in rep:
			rep.append(word)

	##compare results





def text_preprocessing(lyrics):		## lyrics == type: LIST OF VERSES! 

	stemmer = WordNetLemmatizer()
	new_lyrics = []

	for line in lyrics:

		# Remove all the special characters
		new_line = re.sub(r'\W', ' ', str(line))
		

		# remove all single characters
		new_line = re.sub(r'\s+[a-zA-Z]\s+', ' ', str(new_line))

		# Substituting multiple spaces with single space
		new_line = re.sub(r'\s+', ' ', new_line, flags=re.I)

		# Removing prefixed 'b'
		new_line = re.sub(r'^b\s+', '', new_line)

		# Converting to Lowercase
		new_line = new_line.lower()
		
		# Lemmatization
		new_line = new_line.split()  	# we reduce the word into dictionary root form. For instance "cats" is converted into "cat". Lemmatization is done in order to avoid creating features that are semantically similar but syntactically different. For instance, we don't want two different features named "cats" and "cat", which are semantically similar, therefore we perform lemmatization.

		document = [stemmer.lemmatize(word) for word in new_line]
		document = ' '.join(document)
		
		new_lyrics.append(new_line)

	
	#print(new_lyrics)

	return new_lyrics

def get_stopWords(lyrics_tokens,language):

	## NLTK StopWords available in 
	'''
	['hungarian',
	 'swedish',
	 'kazakh',
	 'norwegian',
	 'finnish',
	 'arabic',
	 'indonesian',
	 'portuguese',
	 'turkish',
	 'azerbaijani',
	 'slovene',
	 'spanish',
	 'danish',
	 'nepali',
	 'romanian',
	 'greek',
	 'dutch',
	 'tajik',
	 'german',
	 'english',
	 'russian',
	 'french',
	 'italian']
 	'''
	if language == 'en':
		stopWords = set(stopwords.words('english'))
	elif language == 'es':
		stopWords = set(stopwords.words('spanish'))
	elif language == 'de':
		stopWords = set(stopwords.words('german'))
	elif language == 'du':
		stopWords = set(stopwords.words('dutch'))
	elif language == 'ru':
		stopWords = set(stopwords.words('russian'))
	elif language == 'it':
		stopWords = set(stopwords.words('italian'))
	elif language == 'fr':
		stopWords = set(stopwords.words('french'))
	elif language == 'no':
		stopWords = set(stopwords.words('norwegian'))
	elif language == 'sw':
		stopWords = set(stopwords.words('sweedish'))
	elif language == 'ar':
		stopWords = set(stopwords.words('arabic'))
	else:
		return 0 ## instrumental 

	#print(lyrics_tokens)
	# unnesting verses and estrofas
	lyrics_tokens = sum(lyrics_tokens,[])
	#print(lyrics_tokens)

	stopwords_x = []

	for word in lyrics_tokens:
		if word in stopWords:
			stopwords_x.append(word)

	stopWords_100 =len(stopwords_x) / len(lyrics_tokens) * 100
	
	print(" - % of stopwords ")
	print(stopWords_100) ## porcentage of stop words
	return stopWords_100


def get_language(lyrics): ## lyrics == type: LIST OF VERSES! 

	#print(lyrics)
	language = []
	A = False 

	for verse in lyrics:
		if len(verse)==0 or 'instrumental' in verse.lower() :
			pass
		else:
			A = True
			lang = detect(verse)
			language.append(lang)
	


	if A == True:
		dict_ = collections.Counter(language)
		#print(dict_)

		for key, count in dropwhile(lambda key_count: key_count[1] >= 2, dict_.most_common()):
			del dict_[key]
		print(dict_)
		language = next(iter(dict_))
		if len(dict_)>1:
			lang_mix = 1 
		else:
			lang_mix = 0

	else:
		language = None ## instrumental exception
		lang_mix = 0

	return(language,lang_mix)		## language is for now 'de'/'es'/'en' but this needs to be changed to numerical
									## lang_mix is bool. it determines if a song contains more than one language


## developing:
# song hardcoded -	

#path2song = PATH+'lyrics/Led Zeppelin/Black Dog - Remaster.txt'
#with open(path2song) as f:
#	content = f.readlines()







'''
def init():

	

	path_2_setlist = PATH+'pl_setlist/'
	PL_data = PATH+'PL_DATA/'

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

				###### 
				# if path not exist
				if not os.path.exists(PL_data+user+'/'+pl[:-4]+'/'):
					os.makedirs(PL_data+user+'/'+pl[:-4]+'/')

				# read playlist-setlist
				setlist = pd.read_csv(file,sep='\t')
				print(setlist)
				#sys.exit()
				
				songs_list = setlist['Song1'].tolist()

				del songs_list[0]	## = ID

				artists_list = setlist['Artist1'].tolist()
				del artists_list[0]

				print(songs_list)
				print(artists_list)

				
				for index, song in enumerate(songs_list):
					change_names(song)

				## from get_lyrics
def change_names(song_name)

		if '(' in song_name:
			song_title_w_feat = song_name 
			print(colored('song with parenthesis','red'))
			song_name_ = re.sub(r'\([^)]*\)', '', song_name) ## featuring song-> artist name in parenthesis.
			song_name = song_name_
			print(' Tweaked song name:')
			print(song_name)

		if '-' in song_name: ## = radio edits // remixes - let's get OG
		
			song_name_ = song_name.split('-',1)
			song_name = song_name_[0]
			print(' Tweaked song name:')
			print(song_name)
'''


# init()
#get_lyrics_n_structure()