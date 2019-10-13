import os
import pandas as pd
import numpy as np
from datetime import datetime
from termcolor import colored
import genius_auto
from pathlib import Path


##debugging
import sys

path_ = os.getcwd()


'''
def get_csv(file,user,date,path_day_logs):

    global lyrics_yes
    global lyrics_no

    print("inspecting: "+user+" "+date+" "+file)
    
    df = pd.read_csv(path_day_logs+file)
    #print(df)

    
    song_name_list 	 = df['Song1'].tolist()  ## song_name
    song_id_list     = df['Song2'].tolist()  ## song_id
    artist_name_list = df['Artist1'].tolist()## artist_name 

    del(song_name_list[0])      # = NAME
    del(song_id_list[0])        # = ID 
    del(artist_name_list[0])    # = NAME 

    for indx, song_name in enumerate(song_name_list):
        
        artist_name = artist_name_list[indx]
        print("--- Starting Genius !")
        print("Song Name:   "+song_name)
        print("Artist Name: "+artist_name)


        lyrics, lyrics_bool = genius_auto.init(song_name,artist_name)
        
        if lyrics_bool == 1:
            lyrics_yes +=1
            ## TO DO : save somewhere 
        else:
            lyrics_no  +=1

    sys.exit()
'''


def init():
    start_time = datetime.now()
    lyrics_yes = 0
    lyrics_no  = 0 
    lyrics_done= 0


    all_songs_file  = path_ + '/songs_data/all_songs.csv' 

    lyrics_folder = path_ + '/songs_data/lyrics/'

    if not os.path.exists(lyrics_folder):
        os.mkdir(lyrics_folder)


    no_lyrics_file = path_ + "/songs_data/no_lyrics_in_Genius.txt"
    songs_no_lyrics = []
    
    if not os.path.isfile(no_lyrics_file):
        f = open(no_lyrics_file, 'w')
        f.close()
        
    #  read list from txt ---> OUTPUT: no_lyrics_list 
    else:
        with open(no_lyrics_file,'r') as f:
            for line in f:
                songs_no_lyrics.append(line[:-2])
                print(line[:-2])

    #print(songs_no_lyrics)            

    ##read artist 
    ##read song name + ID 
    ## 

    with open(all_songs_file) as f:
        for i, line in enumerate(f):
            if i == 0: ## = Header
                pass
            else:
                #line = re.split(r'\t+', all_songs_file)
                line = line.split(',')
                print(line)
                if len(line) >> 4:          ## this is bug potential
                    print("----------------------------------------------------")
                    print(" SONG_ID         : " +  line[0])
                    print(" SONG NAME       : " +  line[1])
                    print(" ARTIST NAME     : " +  line[-3])
                    print(" ALBUM NAME      : " +  line[-2])
                    print("----------------------------------------------------")
                #print(i)
                else:
                    print("----------------------------------------------------")
                    print(" SONG_ID         : " +  line[0])
                    print(" SONG NAME       : " +  line[1])
                    print(" ARTIST NAME     : " +  line[2])
                    print(" ALBUM NAME      : " +  line[3])
                    print("----------------------------------------------------")

                song_name = line[1]
                artist_name=line[2]
                album_name =line[3]
                
                elem = artist_name+ " - "+song_name 
                
                ## read and check if already looked for in genius:
                passing = False
                with open(no_lyrics_file,'r') as f:
                    for line in f:
                        #rint(line)
                        #print(elem)
                        if elem+",\n" == line:
                            print(colored("Lyrics for this song could not be found previously...", 'red'))
                            passing = False     ## developing
                        else:
                            passing = False
                
                if passing == False:
                    if '/' in artist_name:
                        print(artist_name)
                        artist_name=artist_name.replace('/','-')

                    if '/' in song_name:
                        print(song_name)
                        song_name = song_name.replace('/','-')
                        #sys.exit(artist_name)

                    ## if song lyrics already in lyrics folder // bypass genius 

                    if os.path.isfile(lyrics_folder+artist_name+'/'+song_name+'.txt'):
                        print(colored("Lyrics for this song already downloaded...", 'green'))
                        print(colored(artist_name+ " -  " + song_name),'cyan')
                        lyrics_done += 1

                    else:

                        ## if song in no_lyrics available .txt

                        lyrics, lyrics_bool = genius_auto.init(song_name, artist_name,album_name)

                        if lyrics_bool == True:
                            lyrics_yes += 1
                            ## create folder 
                            if not os.path.exists(lyrics_folder+artist_name+'/'):
                                os.mkdir(lyrics_folder+artist_name+'/') 

                            with open(lyrics_folder+artist_name+'/'+song_name+".txt", "w") as text_file:
                                text_file.write(lyrics)

                        elif lyrics_bool == False:
                            lyrics_no +=1
                            f = open(no_lyrics_file,'a')
                            f.write(artist_name + " - " + song_name +",\n")
                            f.close()


                        else:
                            sys.exit('Guillermo you suck')
                
    
    print("----------------------------------------------------")
    print("---------------------- TOTAL -----------------------")
    print("----------------------------------------------------")

    time_spent = datetime.now() - start_time
    print(" TIME SPENT TO UPDATE LYRICS     : " + str(time_spent))
    print(" LYRICS ALREADY IN DB            : " + str(lyrics_done))
    print(" LYRICS FOUND	                : " + str(lyrics_yes))
    print(" LYRICS NOT FOUND		        : " + str(lyrics_no))
    


init()
