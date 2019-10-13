## imports
import requests

from termcolor import colored
from bs4 import BeautifulSoup as bs

import re

import urllib

import sys


##############################################
## access token 

clientID = "nirnV08m0NeV9MArdqmw01KngDBmnbfgfTdZTGJjtT0cV-ePMJTr2KFOIzEqvjAp"
clientSecret = "MXT-mfboPvtY3K78B0r225-PyuiD5Ug9r5VzWtuuYXApECZ_-wJi73vsf28hC-7Cz-JpmE5BM4i2dBBR4zW6tw"
token = "NdUWvm-oduXwnXgk5qKhszy6-3S534t93rDwdP2nqA9NynZllsBNBU9ByEUzqjBB"


# TOKEN
parameters = {'client_id':clientID,
            'redirect_uri':"http://localhost:8000/",
            'scope':'',
            'state':'1',
            'response_type':"code"}

genius_auth = "https://api.genius.com/oauth/authorize"


genius_url = "http://api.genius.com"
headers = {'Authorization': 'Bearer '+token}


retryed = False

############## SONG INPUT ##############

#song_name 	= "Get Loose"
#artist_name = "Ms Banks"

############## SONG INPUT ##############


def genius_API(song_api_path):
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


    return song_link



def genius_PARSE(song_link):
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



def lyrics_analysis(lyrics):
    print("")
    print("SONG STRUCTURE")
    song_structure = re.findall("\[([^[\]]*)\]",lyrics)
    print(song_structure)
    
    print("")
    print("SONG COROS")
    song_coros = re.findall('\((.*?)\)',lyrics)
    print(song_coros)

    print("")
    #print("Lyrics blocks")
    lines = lyrics.split('\n')


    ## remove non-lyrics
    for i,line in enumerate(lines):
        if line.startswith('['):
            lines.remove(line) 
            i = i-1
        else:
            if line == '':
                lines[i] = '\n\n'
            else:
                lines[i]=line+('\n')

    lines.remove(lines[0])
    lines.pop()
    lines.remove(lines[0])
    lines.pop()

    print(lines)

    text = ''.join(lines)
    print(text)
    sys.exit()

    indx = [i for i, j in enumerate(lines) if j == '']
    #print(indx)
	
    ### STOPPED DEVELOPING HERE: trying to get a nested list, where every nest is one verse
    verses = []
    i=0
    while i <= len(indx):	
        try:
            if lines[indx[i]+1:indx[i+1]] != []:
                verses.append(lines[indx[i]+1:indx[i+1]])
        except IndexError:
            pass
        i+=1

    print(verses)
    print("")
    print("################### THIS ARE THE VERSES ######################")
    for verse in verses:
        for line in verse:
            print(line)
        print("") 
    print("################### THIS ARE THE VERSES ######################")


def init(song_name, artist_name,album_name):


    if '(' in song_name:
        song_title_w_feat = song_name 
        print(colored('song with parenthesis','red'))
        song_name_ = re.sub(r'\([^)]*\)', '', song_name) ## featuring song-> artist name in parenthesis.
        song_name = song_name_

    search_url = genius_url+'/search'
    search_data = {'q': artist_name + song_name}

    response = requests.get(search_url, params=search_data, headers=headers)
    #print(response)
    json = response.json()

    if len(json['response']['hits'])==0:
        print(colored("No Success in Genius-API...",'red'))
        print("SONG NAME   :                "+song_name)
        print("ARTIST NAME :                "+artist_name)
            
        #print(hit['result']['primary_artist']['name'])
        
        retryed = True                                  ## !! DEVELOPING THIS IS AN ETERNAL LOOP
        #retry(song_name,artist_name,album_name)

        return('error',False)


    for hit in json['response']['hits']:
        #### scraping debugging area ####
        print("####################")

        print(hit['result'])

        print("#####################")

        song_title_hit        =     hit['result']['title']
        song_title_w_feat_hit =     hit['result']['title_with_featured']
        print('----')
        print(song_title_hit)
        print(song_title_w_feat_hit)
        print('----')

        print(hit['result']['primary_artist']['name'])

        #### scraping debugging area ####
        path__ = "/"+artist_name.replace(" ", "-")+"-"+song_name.replace(" ", "-")+"-lyrics"


        if artist_name.lower() in hit['result']['primary_artist']['name'].lower():

            print(colored("Success finding Song in Genius-API",'green'))
            song_info = hit

            song_api_path = song_info['result']['api_path']

            ## CALL FUNCTIONS

            song_link = genius_API(song_api_path)

            lyrics 	  = genius_PARSE(song_link)
			
            #print("type"+str(type(lyrics)))
            #print(len(str(lyrics)))
            print(lyrics)

            #lyrics_analysis(lyrics)

            return(lyrics, True)

        
        elif path__.lower() in hit['result']['path'].lower():

            print(colored("Success finding Song in Genius-API",'green'))
            song_info = hit

            song_api_path = song_info['result']['api_path']

            ## CALL FUNCTIONS

            song_link = genius_API(song_api_path)

            lyrics    = genius_PARSE(song_link)
            
            #print("type"+str(type(lyrics)))
            #print(len(str(lyrics)))
            print(lyrics)

            #lyrics_analysis(lyrics)

            return(lyrics, True)
        else:
            print(colored("No Success in Genius-API...",'red'))
            print("SONG NAME   : 				"+song_name)
            print("ARTIST NAME : 				"+artist_name)
            
            if "-" in song_name and retryed == False:
                print("RETRY")
                song_name_ = song_name.split(' - ')[0]
                retry(song_name_,artist_name,album_name)

            elif "(" in song_name and retryed == False:
                print("RETRY")
                song_name_ = song_name.split('(')[0]
                retry(song_name_,artist_name,album_name)
            
            return('error',False)    
            
            
            

def retry(song_name_, artist_name,album_name):
    if "-" in song_name_ and retryed == False:
        print("RETRY")
        song_name = song_name_.split(' - ')[0]
        

    elif "(" in song_name_ and retryed == False:
        print("RETRY")
        song_name = song_name_.split('(')[0]
        
    init(song_name,artist_name,album_name)


#init(song_name, artist_name)