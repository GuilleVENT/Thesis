import os
import pandas as pd
import genius_
from termcolor import colored



PATH = os.path.dirname(os.path.abspath(__file__))+'/'

path_2_allsongs = PATH+'/songs_data/all_songs.tsv'

all_songs = pd.read_csv(path_2_allsongs,sep='\t')

song_names = all_songs['Song1'].tolist()
artist_names = all_songs['Artist1'].tolist()


for index, song in enumerate(song_names):
	if song == 'Name':
		pass
	else:
		artist = artist_names[index]
		print(colored(' --->\n '+song+' by '+artist,'green'))
		genius_.main(song,artist)
		## CRASH @ Higher Love by Kygo!
		## 

