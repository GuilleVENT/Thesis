
from os import listdir
import os
from pathlib import Path
import pandas as pd
import csv
import sys
from termcolor import colored
import numpy as np




import genius_

from operator import itemgetter

# global result directory of req_spotify.py
path_RES = r'user_PL/'

PATH = os.path.dirname(os.path.abspath(__file__))+'/'

'''
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
						

						data = pd.read_csv(path_2_day+playlist,header=0)


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





def write_link_csv(song_ids,download_link):
	header_songs_ids 	= 'Song_ID'
	header_download_link= 'Download_Link'

	songs_download_link_file = r'songs_data/'

	if os.path.isfile(songs_download_link_file):
		pass
	
	col1 = pandas.DataFrame({header_songs_ids: song_ids})
	col5 = pandas.DataFrame({header_download_link: download_link})
	pandas.concat([df1,df2]).drop_duplicates().reset_index(drop=True)
	##	^^duplicates^			no more haha	
	#df.to_csv(all_songs_logfile,index=False)
	

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
		index_2_delete = check_duplicates(playlist_tuple_2, big_list,ID=1)
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
		index_2_delete = check_duplicates(playlist_tuple,big_list,ID=0)

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
		if ID==1:
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
'''

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


## This part first checks for the most popular playlists of each user 
##  => popular_pl() --> get_followers() --> pl_to_inspect
## then a list of songs in each playlist is written --> to later extract the features of each song individually 
## 

def popular_pl():

	#print(path_RES)

	# 
	## DEVELOPING:
	'''
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
	'''

	path_2_inspect = r'pl_to_inspect/'
	for user in listdir(path_2_inspect):
		if user.startswith('.'):
			pass
		else:
			print(" Path: "+r"pl_to_inspect/"+user)
			#df = pd.read_csv(r'pl_to_inspect/'+user)

			#print(df)
			with open(r"pl_to_inspect/"+user, 'r') as f:
				reader = csv.reader(f)
				pl_list = list(reader)
				pl_list = pl_list[0]

				for pl in pl_list:
					if len(pl)==0 or pl.startswith('.'):
						pass
					else:
						get_songs(user, pl)

	#sys.exit()

def get_songs(user,pl):
	print(user[:-4])
	user = user[:-4]
	
	print(pl)
	#sys.exit()

	for day in listdir(path_RES+user):
		if not day.startswith('.'):
			#print(day)

			
			#print(pl)
			#print(day)
			#print(user)
			#print(type(pl))
			#print(type(day))
			#print(type(user))
			#sys.exit()
			path_2_day = user+'/'+day+'/'+pl+'.csv'
							
			print(" ***** PLAYLIST = "+colored(pl,'green')+" *****")
			print(path_2_day)
			
			try:
				data = pd.read_csv(PATH+path_RES+path_2_day,header=0)
				df = data[['Song2','Artist1','Song1','Preview_URL']]

				print(df)
				pl_setlist(user,pl,df)

			except FileNotFoundError:
				print('Error: Could not find '+colored(path_2_day,'red'))

			
			#sys.exit()
			#song_names   = data.Song1.values.tolist()
			#artist_names = data.Artist1.values.tolist()
			#song_ids     = data.Song2.values.tolist()
			#download_link= data.Preview_URL.values.tolist()

			#print(song_names)
			#print(artist_names)
			#print(song_ids)
			#print(download_link)
			

			
			## i'm going to write a TSV file so no check for commas is needed
			'''
			for counter, song_name in enumerate(song_names):
				if "," in song_name:
					#print(song_name)
					#print(colored("HERE",'blue'))
					song_names[counter] = song_name.replace(',',' ')

			for counter, artist_name in enumerate(artist_names):
				if "," in artist_name:
					artist_names[counter] = artist_name.replace(",",' ')
			'''

			#song_names.pop(0)
			#artist_names.pop(0)
			#song_ids.pop(0)
			#download_link.pop(0)
			#sys.exit()


def pl_setlist(user,pl,df):

	path_2_playlist_setlist = r'pl_setlist/'
	path = path_2_playlist_setlist+user+'/'
	#print(os.path)
	if not os.path.exists(path):
		os.makedirs(path)

	file = Path(PATH+path+pl+'.tsv')
	#print(file)
	if not file.exists():
		df.to_csv(file,index=False,sep='\t')

	else:
		print(file)
		read = pd.read_csv(file,sep='\t')
		# dropping ALL duplicte values 
		res  = pd.concat([read,df])
		
		res.drop_duplicates(subset='Song1',keep = 'first', inplace = True)
		
		result_df = pd.concat([read,res])
		##	^^duplicates^			no more haha	
		result_df.to_csv(file,sep='\t',index=False)

	print(' -> done with setlist \n all songs data...')
	write_all_songs(user,pl,df)

def write_all_songs(user,pl,df):
	
	all_songs_file = Path(r'songs_data/all_songs.tsv')
	if not all_songs_file.exists():
		df.to_csv(all_songs_file,index=False,sep='\t')
	
	else:
		read = pd.read_csv(all_songs_file,sep='\t')
		all_songsID = read.Song1.values.tolist()
		# dropping ALL duplicte values 
		'''res  = pd.concat([read,df])
		print(res)
		print(' - - - ')
		da=res.drop_duplicates(subset='Song1',keep = 'first', inplace = False)
		print(da)

		result_df = pd.concat([read,da])
		print(result_df)
		##	^^duplicates^			no more haha
		'''
		for index, row in df.iterrows():
			if row['Song1'] in all_songsID:
				df=df.drop(index)
			else:
				pass

		res = pd.concat([read,df])
		
		res.to_csv(all_songs_file,sep='\t',index=False)
		


def get_followers(user,days,pl_list):

	print(" **************************************** ")
	
	pl_follow_avrg = []
	
	for pl in pl_list:
		print(" PLAYLIST: "+colored(pl,'blue'))
		days_column = ['Days']
		popularity_column = ['Followers']
		pop = []

		sorting_list = []
		
		i = 0
		
		while i < len(days):
			path = path_RES+user+'/'+days[i]+'/'+pl
			print(" "+colored(path,'cyan'))
			try:
				df = pd.read_csv(path)
				popularity = int(df['Popularity'].iloc[0])
			except IOError:
				print(" FAIL FINDING PL:"+colored('ERROR--Code:1','red'))
				print(user)
				print(pl)
				fail_path = r"pl_popularity/deleted_pl.txt"
				with open(fail_path,'a') as fails:				## this should be emptied before the function get followers is called. BC it just appends the same stuff over and over
					fails.write(user+' , '+pl+' // ')
				break
			except ValueError:
				print(" FAIL FINDING PL:"+colored('ERROR--Code:2','red'))
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

			month, day = days[i].split('_')


			sorting_list.append(tuple((day,month,popularity)))
			sorted_list = sorted(sorting_list, key=lambda element: (int(element[0]), int(element[1])))



			i = i+1

		#sys.exit(sorted_list)

		df = pd.DataFrame(sorted_list, columns=['Day','Month','Followers'])
		output_path = r'pl_popularity/'+user+'/'
		
		if not os.path.exists(output_path):
			os.makedirs(output_path)
		
		df.to_csv(output_path+pl)


		
		avrg_pop = df['Followers'].mean()

		temp_tuple = (pl[:-4],avrg_pop)

		pl_follow_avrg.append(temp_tuple)

		
	

		#print(days_column)
		#print(popularity_column)
		#print(type(days_column))
		#print(type(popularity_column))
		#sys.exit(output_path+pl)

		#with open(output_path+pl,'w') as f:
		#	df.to_csv(index=False)

		
	
	print(" *********************** ")
	print(pl_follow_avrg)
	print(" *********************** ")

	pl_to_inspect(user,pl_follow_avrg)
	

def pl_to_inspect(user,pl_follow_avrg):

	## reducing number of users:
	## change this to reduce the number of users to inspect! 
	if user == 'spotify':
		Num = 10
	else:
		Num = 5

	followers_l_sorted = sorted(pl_follow_avrg, key=lambda tup: tup[1])

	followers_tupl = followers_l_sorted[::-1]

	#print("PL of user "+user+" sorted:")
	#print(followers_tupl)
	#sys.exit()

	top = followers_tupl[:Num]		### GET TOP 15 PL
									### top is a variable of the top X pl with the most followers

	path_csv_file =  r"pl_to_inspect/"
	if not os.path.exists(path_csv_file):
			os.makedirs(path_csv_file)

	path_csv_file = open(path_csv_file+""+user+".csv",'w')

	for elemnt in top:
		path_csv_file.write("%s," %elemnt[0])



def init():

	popular_pl()


	# input these popular playlist to write these files:
	#write_all_songs()
	
	#get_all_songs()   ## gathers all the data of the songs of the popular playlists! 


init()


