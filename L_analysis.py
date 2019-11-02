import os
import pandas as pd
import sys
from termcolor import colored




PATH = os.path.dirname(os.path.abspath(__file__))+'/'



# developing:
# song hardcoded -
def get_lyrics_n_structure(file):

	## developing:
	# song hardcoded -
	path2song = PATH+'lyrics/Led Zeppelin/Black Dog - Remaster.txt'

	## 											--> get_lyrics changes the names of the files to not contain "remaster"
	



	with open(path2song) as f:
		content = f.readlines()

	lyrics = [x.strip() for x in content] ## each line is an element in a list

	structure = []

	for line in lyrics:
		if line.startswith('['):
			structure.append(line)
			lyrics.remove(line)

	print(lyrics)

	print(structure)




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



init()