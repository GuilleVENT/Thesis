import requests
import base64
import json

clientID     =  "fa83a2fec7c440b7a2795d600d6b2c7e"
clientSecret =  "4fa07375d53447aeb3482a716a359587"


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