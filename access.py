import requests
import base64
import json

## this is given to you when you register on Spotify's API
clientID     =  ""
clientSecret =  ""
## edit it

def get_token():
    payload = clientID + ":" + clientSecret

    encodedPayload = base64.b64encode(payload.encode())

    url = "https://accounts.spotify.com/api/token"
    body = {"grant_type":"client_credentials"}

    headers ={
     "Authorization": "Basic " +encodedPayload.decode(),
      }

    response = requests.request("POST", url, data=body, headers=headers)
    print(response)
    
    #print(response.text)

    data = json.loads(response.text)
    
    access_token = data['access_token']
    print("Access Token:"+str(access_token))
    return access_token


#if no mothership :
#get_token()