from selenium import webdriver
import time
from bs4 import BeautifulSoup
from urllib.request import urlopen, re

########### 전일 거래량 20종목 가져오기 ############

def getStocks(Path):    
    # 주식수 (20개)
    stockNum = 20

    # 해당 페이지가 주식 정보를 동적으로 만들어 selenium으로 데이터를 가져온다.
    options = webdriver.ChromeOptions()
    options.add_argument('headless')

    # driver = webdriver.Chrome('chromedriver.exe', options=options)
    driver = webdriver.Chrome(Path, options=options)


    ####################### 종목명, 종목정보 링크(etf종목 구별할 때 사용) #################

    driver.get('https://finance.naver.com/sise/sise_quant.naver')

    time.sleep(5)
    bs = BeautifulSoup(driver.page_source, 'html.parser')

    # 테이블에서 PER|ROE|영업이익(etf,etn면 N/A) 위치
    index = 0 
    ths = bs.find(class_='type_2').find('tbody').find('tr').find_all('th')
    for i in range(len(ths)):
        if(ths[i].get_text() == 'PER' or ths[i].get_text() == 'ROE' or ths[i].get_text() == '영업이익'):
            index = i+1
            break
            
    
    # 테이블의 종목당 정보들 
    stocks = bs.find(class_='type_2').find('tbody').find_all('tr')

    result = []

    count = 0
    # count가 주식 수를 채울 때 까지 반복
    for i in range(len(stocks)):
        
        # 테이블에서 종목당 종목 정보만 가져오기
        stock = None
        try:
            no = stocks[i].find(class_="no").get_text()
            stock=stocks[i]
        except AttributeError:
            continue
        
        # PER|ROE|영업이익 N/A가 아니면 종목명 리스트에 넣기
        if(stock.select('td:nth-child({})'.format(index))[0].get_text() != 'N/A'):
            result.append(stock.find(class_="tltle").get_text()) 
            count += 1
        
        # 원하는 종목 만큼 (stockNum == 20)
        if(count>=stockNum):
            break

    driver.quit()

    return result