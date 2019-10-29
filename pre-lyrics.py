import os
import pandas as pd
import genius_
from termcolor import colored



PATH = os.path.dirname(os.path.abspath(__file__))+'/'

path_2_allsongs = PATH+'songs_data/all_songs.tsv'

path_2_lyrics = PATH+'lyrics/'


all_songs = pd.read_csv(path_2_allsongs,sep='\t')




song_names = all_songs['Song1'].tolist()
artist_names = all_songs['Artist1'].tolist()






for index, song in enumerate(song_names):
	artist = artist_names[index]
	if song == 'Name':
		pass

	elif os.path.isfile(path_2_lyrics+artist+'/'+song+'.txt'):
		pass
	
	else:
		print(colored(' --->\n '+song+' by '+artist,'green'))
		genius_.main(song,artist)
		# debugging:
		#genius_.main('Higher Love','Kygo')
		# CRASH @ Higher Love by Kygo!
		## 

