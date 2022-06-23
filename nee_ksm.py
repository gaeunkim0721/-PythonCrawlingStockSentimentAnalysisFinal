from fileinput import filename
import requests
import json
from datetime import date


#인가코드 받기 
# REST API : '772e5236d7a793ed613505af7b6beb15'
#https://kauth.kakao.com/oauth/authorize?response_type=code&client_id=772e5236d7a793ed613505af7b6beb15&redirect_uri=https://example.com/oauth



url = 'https://kauth.kakao.com/oauth/token'
rest_api_key = '772e5236d7a793ed613505af7b6beb15'
filename = "./kakao_code.json"

#최초 세팅.. token_refresh 안될때 
def set_tokens():
    redirect_uri = 'https://example.com/oauth'
    authorize_code = 'Z7S4d-N7vmdBnv0wdYzIxxaWRB3OnogKPXwkq7Xt9RAXXZnSI-iKjxCNjWYpeVrB0XA4Xgopb1UAAAGBjfQIsA'

    data = {
        'grant_type':'authorization_code',
        'client_id':rest_api_key,
        'redirect_uri':redirect_uri,
        'code': authorize_code,
    }

    response = requests.post(url, data=data)
    tokens = response.json()
    save_tokens(tokens)
    print(tokens)

    return tokens

#토큰 저장
def save_tokens(tokens):
    with open(filename, "w") as fp:
        json.dump(tokens, fp)


def load_tokens():
    with open(filename) as fp:
        tokens = json.load(fp)
    return tokens


def token_refresh():
    loadtk = load_tokens()
    
    url = "https://kauth.kakao.com/oauth/token"
    data = {
        "grant_type": "refresh_token",
        "client_id": rest_api_key,
        "refresh_token": loadtk["refresh_token"]
    }
    
    response = requests.post(url, data=data)
    
    loadtk['access_token'] = response.json()['access_token']
    save_tokens(loadtk)
    
    return loadtk



nee_template_id = 78646
date = str(date.today().strftime("%Y년 %m월 %d일".encode('unicode-escape').decode()
    ).encode().decode('unicode-escape')
)

def sendmsg(token, resultsList):

    url="https://kapi.kakao.com/v2/api/talk/memo/send"

    # kapi.kakao.com/v2/api/talk/memo/send

    headers={
        "Authorization" : "Bearer " + token["access_token"]
    }
    
    msglist =[["" for col in range(2)] for row in range(5)]
    # for i, (k , v) in enumerate(resultsDict.items()):

    for i in range(len(resultsList)):
        rtuple = resultsList[i]
        theader = str(i+1)+"위"
        tbody = rtuple[1] +"("+ str(rtuple[0]) +"점)"
        msglist[i]=[theader, tbody]

    
    data={
        "template_id" : nee_template_id,
        "template_args" : json.dumps({
        "${TOP1_H}" : msglist[0][0],
        "${TOP1_B}" : msglist[0][1],
        "${TOP2_H}" : msglist[1][0],
        "${TOP2_B}" : msglist[1][1],
        "${TOP3_H}" : msglist[2][0],
        "${TOP3_B}" : msglist[2][1],
        "${TOP4_H}" : msglist[3][0],
        "${TOP4_B}" : msglist[3][1],
        "${TOP5_H}" : msglist[4][0],
        "${TOP5_B}" : msglist[4][1],
        "${TODAY}" : date,
        }),
    }

    response = requests.post(url, headers=headers, data=data)
    print(response.status_code)
    print(response.text)




