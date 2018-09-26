import time
import numpy as np
from bs4 import BeautifulSoup as bs
import requests
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys


browser = webdriver.Chrome()
url = u'https://twitter.com/karpathy'

browser.get(url)
time.sleep(1)


body= browser.find_element_by_tag_name('body')
for _ in range(200):
    body.send_keys(Keys.PAGE_DOWN)
    time.sleep(0.2)


html_source=browser.page_source
sourcedata= html_source.encode('utf-8')
soup=bs(sourcedata, 'html.parser')
arr = soup.body.findAll('a', class_='account-group js-account-group js-action-profile js-user-profile-link js-nav')
pclass = soup.body.find_all('p', class_='TweetTextSize TweetTextSize--normal js-tweet-text tweet-text')


f1 = open('word.text', 'a')
for ix in range(200):
    try:
        user_id = arr[ix].attrs['href']  
    except IndexError:
        break
    twit = pclass[ix].text
    f1.write("{}: {}".format(user_id[1:], twit) + "\n")
    f1.write('\n')
    
f1.close()

