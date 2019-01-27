import requests
import json
url = 'http://127.0.0.1:8000/api-token-auth/'
#headers = {'Authorization': 'Token 5be1e7c4437cfa2c61f77347430c72e73b0fa69f'}


data = {'username':'car','password':'car123456'}
r = requests.post(url,json=data,headers={'Content-Type': 'application/json' })
if r.status_code != 200:
    raise ApiError('POST /api-token-auth/ {}'.format(r.status_code))
token = r.json()['token']

url = 'http://127.0.0.1:8000/signs/toupdate'
headers = {'Authorization': 'Token '+token}
r = requests.get(url, headers=headers)
for item in r.json():
    url='http://127.0.0.1:8000/signs/'+str(item['id'])+'/'

    data = {'id':item['id'],'name':item['name'],'latitude':item['latitude'],
    'longitude':item['longitude'],'country':item['country'],'county':item['county'],
    'neighbourhood':item['neighbourhood'],'road':item['road'],'speedlimit':item['speedlimit'],
    'is_uploaded':True}
    r = requests.put(url,json=data,headers={'Content-Type': 'application/json',
     'Authorization': 'Token '+token})
    print(r.content)