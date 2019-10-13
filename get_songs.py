
from os import listdir
import os
from pathlib import Path
import pandas
import csv
import sys
from termcolor import colored
import numpy as np


import genius_

from operator import itemgetter


def write_all_songs():
	path_RES = r'user_PL/'
	days = []

	#				***DEVELOPING***
	
	for user in listdir(path_RES):
		if not user.startswith('.'):
			print(" ***** USER = "+colored(user,'red')+" *****")
			#print(user)

			## part 1 get all songs

			for day in listdir(path_RES+user):
				if not day.startswith('.'):
					#print(day)

					path_2_day = path_RES+user+'/'+day+'/'
					print(path_2_day)

					for playlist in listdir(path_2_day):
						
						print(" ***** PLAYLIST = "+colored(playlist,'green')+" *****")
						print(path_2_day+playlist)
						

						data = pandas.read_csv(path_2_day+playlist,header=0)


						song_names   = data.Song1.values.tolist()
						artist_names = data.Artist1.values.tolist()
						song_ids     = data.Song2.values.tolist()

						#print(song_names)
						#print(artist_names)
						#print(song_ids)


						for counter, song_name in enumerate(song_names):
							if "," in song_name:
								#print(song_name)
								#print(colored("HERE",'blue'))
								song_names[counter] = song_name.replace(',',' ')

						for counter, artist_name in enumerate(artist_names):
							if "," in artist_name:
								artist_names[counter] = artist_name.replace(",",' ')


						song_names.pop(0)
						artist_names.pop(0)
						song_ids.pop(0)

						#print(song_names)
						#print(artist_names)
						#print(song_ids)

						if len(song_names) == 0:
							pass
						else:
							
							playlist_tuple = list(zip(artist_names, song_names))
							#print(playlist_tuple)
							#sys.exit()
							#check_n_write(playlist_tuple,ID=False)
							

							playlist_tuple_2 = list(zip(artist_names,song_names,song_ids))
							#print(playlist_tuple_2)
							#sys.exit()
							check_n_write(playlist_tuple_2,ID=True)
'''	
						#song_names   = data.Song1.values.tolist()
						#artist_names = data.Artist1.values.tolist()
						#album_names  = data.Album1.values.tolist()
						#download_link= data.Preview_URL.values.tolist()


						## *** POPULARITY ***
						## read playlists followers
						#pop_col = data.Popularity.values.tolist()
						#followers = pop_col.pop(0)
						## read playlists followers
						#get_popularity(song_ids, song_names, artist_names, album_names, pop_col)



						#header_songs_ids 	= str('Song_'+song_ids.pop(0))
						#header_songs_names 	= str('Song_'+song_names.pop(0))
						#header_artist    	= str('Artist_'+artist_names.pop(0))
						#header_album 		= str('Album_'+album_names.pop(0))
						#download_link.pop(0)
						#header_download_link= str('Download_Link')
						
						
						#col1 = pandas.DataFrame({header_songs_ids: song_ids})
						#col2 = pandas.DataFrame({header_songs_names: song_names})
						#col3 = pandas.DataFrame({header_artist: artist_names})
						#col4 = pandas.DataFrame({header_album: album_names})
						#col5 = pandas.DataFrame({header_download_link: download_link})

						#df = pandas.concat([col1,col2,col3,col4,col5], axis=1)
						#print(df)
						
						##no need to check for duplicates
						#df.to_csv(all_songs_logfile,index=False)
'''						#df.to_csv(all_songs_logfile_tsv,index=False,sep='\t')



def check_n_write(playlist_tuple,ID):
	## i can be 1 or 0 

	if ID == True:
		t = "ID"	 ## TRUE
		playlist_tuple_2 = []
		for song in playlist_tuple:
			playlist_tuple_2.append(tuple(reversed(song)))
			#print(playlist_tuple_2)
			# -

	
	else:
		t = ""

	#print(playlist_tuple_2)
	#sys.exit('here')

	songs_data = r'songs_data/'
	all_songs_txt = songs_data+"all_songs"+t+".txt"

	big_list = []

	if not os.path.exists(songs_data):
		os.makedirs(songs_data)

	if not Path(all_songs_txt).exists():
		open(all_songs_txt, 'a').close()
	else:
		
		## READ existing list:
		
		with open(all_songs_txt, "rb") as fp:
			for i in fp.readlines():
				tmp = i.decode().split(",")
				#print(tmp)
				
				#sys.exit()
				## variation
				if ID == True:
					#print((tuple((tmp[0],tmp[1],tmp[2][:-2]))))
					big_list.append(tmp[0]) 
					#### developing HERE !! Saturday Night. 11.10 --> solved

				else:
					big_list.append(tuple((tmp[0], tmp[1][1:][:-2])))
				#print(big_list)

	if ID == True:
		index_2_delete = check_duplicates(playlist_tuple_2, big_list,ID=True)
		#print(playlist_tuple_2)
		#print(index_2_delete)

		if len(index_2_delete) == len(playlist_tuple_2):
			print("====>"+colored("All the songs in this playlist were already in all_songs"+t+".txt",'cyan'))
			return

		
		elif len(index_2_delete) != 0:
			for indx in index_2_delete[::-1]:
				#print(playlist_tuple_2)
				#print(index_2_delete)
				#print(playlist_tuple_2[indx])
				del playlist_tuple_2[indx]
				#sys.exit(playlist_tuple_2)
		
		
		for elem in playlist_tuple_2:
			print(elem[0],elem[1],elem[2])
			if elem[0] == float('nan'):
				pass
			else:
				line = elem[0]+', '+elem[1]+', '+elem[2]+' \n'
				print(" + + + + + + + + THIS IS THE LINE TO BE WRITTEN + + + + + + + + + + +")
				print(colored(line,'blue'))
				#sys.exit('debugging')
				with open(all_songs_txt, "a") as file:
					file.write(line)

	else:
		index_2_delete = check_duplicates(playlist_tuple,big_list,ID=False)

		if len(index_2_delete) == len(playlist_tuple):
			print("====>"+colored("All the songs in this playlist were already in all_songs.txt",'cyan'))
			return 


		elif len(index_2_delete) != 0:
			for indx in index_2_delete[::-1]:
				del playlist_tuple[indx]

			
		for elem in playlist_tuple:
			line = elem[0]+", "+elem[1]+" \n"
			print(" + + + + + + + +/ THIS IS THE LINE TO BE WRITTEN /+ + + + + + + + + +")
			print(colored(line,'blue'))
			with open(all_songs_txt, "a") as file:
				file.write(line)




def check_duplicates(playlist_tuple,big_list,ID):
	l_indx = []

	for index, elem in enumerate(playlist_tuple):
		if ID==True:
			song_id = elem[0]
			song_name = elem[1]
			artist_name = elem[2]
			#print(song_id)
			#print(song_name)
			#print(artist_name)

			if song_id in big_list:
				l_indx.append(index)
				print('DUPLICATE:')
				print(colored("--->",'red')+" "+song_id+", "+song_name+", "+artist_name)

		else:
			artist = elem[0]
			song   = elem[1]


			#print(artist+", "+song)

		
			if elem in big_list:
				l_indx.append(index)
				print('DUPLICATE:')
				print(colored("--->",'red')+" "", "+song+", "+artist)

	#print(l_indx)
	return l_indx


def get_all_songs():

	songs_data = r'songs_data/'
	all_songs_txt = songs_data+"all_songs.txt"

	with open(all_songs_txt) as f:
		lines = f.readlines()

	#print(line)
	#print(type(line))

	for song in lines:
		[artist_N, song_N] = song.split(',')
		song_N = song_N[1:][:-2]
		
		print(song_N)
		print(artist_N)
		genius_.main(song_N,artist_N)






def init():
	write_all_songs()
	#get_all_songs()


init()





''' vvv---> these are the rests <---vvv '''
'''	for elem in playlist_tuple:
	print(elem)

		if len(big_list) == 0:
			print(colored('all_songs.txt appears to be empty!','magenta'))
		

			line = elem[0]+", "+elem[1]+" \n"
			print(line)
			with open(all_songs_txt, "a") as file:
				file.write(line)
'''
'''
	for elem in playlist_tuple:
		big_list.append(elem)
		big_set = set([i for i in big_list])

	for elem in big_set:
		line = elem[0]+", "+elem[1]+" \n"
		print(" + + + + + + + + THIS IS THE LINE TO BE WRITTEN + + + + + + + + + + +")
		print(line)
		with open(all_songs_txt, "a") as file:
			file.write(line)
'''
'''
		else:
			for item in  big_list:
				#print(item)
				#print(type(item))

				if elem[0] == item[0] and elem[1] == item[1]:
			
					print(" --------------- "+colored("this item was already in the list: ",'cyan')+" --------------- ")
					print(elem[0])
					print(elem[1])
					print(item)
					print(" --------------- ")

					# delete elem from list
					# playlist_tuple.remove(elem)
					pass
				
			

			line = elem[0]+", "+elem[1]+" \n"
			print(" + + + + + + + + THIS IS THE LINE TO BE WRITTEN + + + + + + + + + + +")
			print(line)
			with open(all_songs_txt, "a") as file:
				file.write(line)
			pass
'''