
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

# global result directory of req_spotify.py
path_RES = r'user_PL/'


def write_all_songs():
	
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

						download_link= data.Preview_URL.values.tolist()

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
						download_link.pop(0)

						#print(song_names)
						#print(artist_names)
						#print(song_ids)

						if len(song_names) == 0:
							pass
						else:
							
							playlist_tuple = list(zip(artist_names, song_names))
							#print(playlist_tuple)
							#sys.exit()
							check_n_write(playlist_tuple,ID=0)
							

							playlist_tuple_2 = list(zip(artist_names,song_names,song_ids))
							#print(playlist_tuple_2)
							#sys.exit()
							check_n_write(playlist_tuple_2,ID=1)

							#write_link_csv(song_ids,download_link)


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
						#####download_link.pop(0)
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



def write_link_csv(song_ids,download_link):
	header_songs_ids 	= 'Song_ID'
	header_download_link= 'Download_Link'

	songs_download_link_file = r'songs_data/'

	if os.path.isfile(songs_download_link_file):
		pass
	'''
	col1 = pandas.DataFrame({header_songs_ids: song_ids})
	col5 = pandas.DataFrame({header_download_link: download_link})
	pandas.concat([df1,df2]).drop_duplicates().reset_index(drop=True)
	##	^^duplicates^			no more haha	
	#df.to_csv(all_songs_logfile,index=False)
	'''

def check_n_write(playlist_tuple,ID):
	## i can be 1 or 0 

	if ID == 1:
		t = "ID"	 ## TRUE
		playlist_tuple_2 = []
		for song in playlist_tuple:
			playlist_tuple_2.append(tuple(reversed(song)))
			#print(playlist_tuple_2)
			# -

	
	if ID == 0:
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
			#
			if type(elem[0]) == type(1.0):
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
				#print('DUPLICATE:')
				#print(colored("--->",'red')+" "+song_id+", "+song_name+", "+artist_name)

		else:
			artist = elem[0]
			song   = elem[1]


			#print(artist+", "+song)

		
			if elem in big_list:
				l_indx.append(index)
				#print('DUPLICATE:')
				#print(colored("--->",'red')+" "", "+song+", "+artist)

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

def popular_pl():

	#print(path_RES)

	# 
	for user in os.listdir(path_RES):
		if user.startswith('.'):
			pass
		else:
			print(path_RES+user)
			
			days = os.listdir(path_RES+user)
			pl_list = os.listdir(path_RES+user+"/"+days[1])

			for i, day in enumerate(days):
				if day.startswith("."):
					del days[i]

			for i, pl in enumerate(pl_list):
				if pl.startswith('.'):
					del pl_list[i]

			get_followers(user, days, pl_list)

def get_followers(user,days,pl_list):

	print(" **************************************** ")
	
	pl_follow_avrg = []
	
	for pl in pl_list:
		days_column = ['Days']
		popularity_column = ['Followers']
		pop = []
		
		i = 0
		
		while i < len(days):
			try:
				path = path_RES+'/'+user+'/'+days[i]+'/'+pl
				df = pandas.read_csv(path)
				popularity = int(df['Popularity'].iloc[0])
			except IOError:
				print(" FAIL FINDING PL:")
				print(user)
				print(pl)
				fail_path = r"pl_popularity/deleted_pl.txt"
				with open(fail_path,'a') as fails:				## this should be emptied before the function get followers is called. BC it just appends the same stuff over and over
					fails.write(user+' , '+pl+' // ')
				break
			except ValueError:
				print(" FAIL FINDING PL:")
				print(user)
				print(pl)
				fail_path = r"pl_popularity/deleted_pl.txt"
				with open(fail_path,'a') as fails:				## this should be emptied before the function get followers is called. BC it just appends the same stuff over and over
					fails.write(user+' , '+pl+' // ')
				break

			popularity_column.append(popularity)
			#print popularity_column
			#print popularity
			#print type(popularity)
			pop.append(popularity)
			days_column.append(days[i])


			i = i+1

		if len(pop)!=0:
			avrg_pop = sum(pop) / float(len(pop))
			temp_tuple = (pl[:-4],avrg_pop)

			pl_follow_avrg.append(temp_tuple)

			output_path = r'pl_popularity/'+user+'/'
			if not os.path.exists(output_path):
				os.makedirs(output_path)

			print(days_column)
			print(popularity_column)
			print(type(days_column))
			print(type(popularity_column))
			#sys.exit(output_path+pl)

			with open(output_path+pl,'w') as f:
				writer = csv.writer(f)
				writer.writerows(zip(days_column,popularity_column))
		
		
	print(" *********************** ")
	print(pl_follow_avrg)
	print(" *********************** ")

	get_popular_pl(user,pl_follow_avrg)


def get_popular_pl(user,pl_follow_avrg):

	followers_l_sorted = sorted(pl_follow_avrg, key=lambda tup: tup[1])

	followers_tupl = followers_l_sorted[::-1]

	print("PL of user "+user+" sorted:")
	print(followers_tupl)

	top = followers_tupl[:15]		### GET TOP 15 PL
									### top is a variable of the top X pl with the most followers

	path_csv_file =  r"pl_to_inspect/"
	if not os.path.exists(path_csv_file):
			os.makedirs(path_csv_file)

	path_csv_file = open(path_csv_file+"pl_to_inspect_"+user+".csv",'w')
	for elemnt in top:
		path_csv_file.write("%s," %elemnt[0])



def init():

	popular_pl()

	# input these popular playlist to write these files:
	write_all_songs()
	
	#get_all_songs()   ## gathers all the data of the songs of the popular playlists! 


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