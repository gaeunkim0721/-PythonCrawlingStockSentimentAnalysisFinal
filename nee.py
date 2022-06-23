import csv
import json
import os,re
import sys
import urllib.request
from openpyxl import Workbook
from datetime import datetime, timedelta, date

# 네이버 api id password
NEE_CLIENTID = "Y8J4YHW3zwtFxu4WcFvs"
NEE_CLIENTPW = "CG9dbamsK4"


# 날짜(기간)를 구하는 함수!
def dateList(): 
    global datelist
    datelist = []
    for i in range(7): #7일간의 기사를 가져와야 하기 때문에 날짜 데이터를 리스트에 담습니다.
        datecalc = datetime.today() - timedelta(i)
        checkDate = str(datecalc.strftime("%d %b %Y")) #날짜 포멧 만들기 일 월(영문) 년
        datelist.append(checkDate)


#엑셀로 저장하는 함수! 
# 불러오기 하면 깨져서 사용안하기로함 
def makeexcel(headline):
    
    #파일이름 만들기 기사모음_오늘날짜.xlsx
    today = date.today()
    today = re.sub("-", "", str(today.isoformat()))
    filename = "기사모음_"+today+".xlsx"
    
    wbook = Workbook()
    
    wactive = wbook.active
    wactive.title = "뉴스기사"
    
    wactive.append(["종목이름","기사제목"]) #컬럼 이름 입니다. 
    
    for line in headline: # 한줄씩
        wactive.append(line) # 넣어줍니다.
    
    wbook.save(filename)
    wbook.close
    return print("엑셀저장 완료!") 

#csv 로 저장 
def makecsv(headline):
    today = date.today()
    today = re.sub("-", "", str(today.isoformat()))
    filename = "기사모음_"+today+".csv"
    
    with open(filename, "w", encoding="utf-8", newline='') as fs:
        wr = csv.writer(fs)
        wr.writerow(["종목이름","기사제목"])
        for line in headline:
            wr.writerow(line)
            
    return filename


# 헤드라인 가져오기 
def getHeadline(stock):
    titlelist=[]
    
    header = {
        "X-Naver-Client-Id" : NEE_CLIENTID,
        "X-Naver-Client-Secret" : NEE_CLIENTPW
    }
    encText = urllib.parse.quote(stock) 

    for i in range(0,1000,100): 
        if len(titlelist) >= 100: break 
        count = 1 + i
        params = "?query=" + encText + "&sort=sim&start="+ str(count) +"&display=100&" #관련도순, count부터 count+100개 가져옵니다
        url = "https://openapi.naver.com/v1/search/news.json"+params 

        request = urllib.request.Request(url, headers=header)
        response =urllib.request.urlopen(request)
        rescode = response.getcode()
        

        if(rescode==200):
            response_body = response.read()
            results = json.loads(response_body.decode('utf-8'))
            
            items = results["items"] # itmes는 딕셔너리를 값으로 가진 리스트입니다.
            
            for dicts in items: #딕셔너리를 하나씩 꺼냅니다.
                innerList = []
                innerList.append(stock)
                titles = filter(dicts) # 뉴스 헤드라인을 다듬습니다.
                if titles != "" : #비어있는것이 아니라면 리스트에 담습니다. 
                    if len(titlelist) >= 100 : break # 100개를 넘었다면 끝냅니다. 
                    innerList.append(titles) 
                    titlelist.append(innerList)
                else :
                    pass            
        else:
            print("Error Code:" + str(rescode))
            
    return titlelist

# 날짜 확인, 필요없는 문자 제거하는 함수 
def filter(items):
    tt = ""
    for day in datelist:
        if day in items["pubDate"]: #pubDate는 게시된 날짜가 들어있습니다. 날짜값을 조회힙니다.
            #key 이름이 title인 value에 헤드라인이 들어있습니다. 
            tt = items["title"].replace("<b>","") # 필요없는 문자를 제거합니다(1)
            tt = tt.replace("</b>","") # 필요없는 문자를 제거합니다(2)
            tt = re.sub("quot;","",tt) # 필요없는 문자를 제거합니다(3)
            tt = re.sub("amp;","",tt) # 필요없는 문자를 제거합니다(4)
            break
        else: pass

    return tt 


def start(stocks):
    dateList() # 날짜리스트 구하기 
    headlines = [] # headlines 안에 리스트로 ["종목이름","뉴스헤드라인"] 값이 들어갑니다. 
    for stk in stocks:
        headlines += getHeadline(stk)

    filename = makecsv(headlines)
    return filename

