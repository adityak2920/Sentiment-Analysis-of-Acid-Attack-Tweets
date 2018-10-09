import time
import numpy as np
from bs4 import BeautifulSoup as bs
import requests
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys


browser = webdriver.Chrome()
url = u'https://twitter.com/michael_nielsen'

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
fo

##the method which is coded above using selenium and bs4 is quite slow so it will take a lot of time to scrape around 20000 tweets it's performs fairly well for small no. of tweets. For this project i had used 'twint' github repo
##Using twint i can scrape 10000 tweets in less than 15 minutes it's pretty faster than selenium method
##when using twint i don't need to write any code the steps for this are:
###1. clone the github repo of twint
###2. now open terminal and go to twint repo
###3. then with one line of command i can scrape and save in csv format tweets so, the command for t tweets containg word 'acid attack threat' i just need to write:
####  'python3 Twint.py -s 'acid attack threat' -o killacidindia.csv --csv'
### so it's much more simple and it gives csv with tweets and all other information about tweets like user id, username, tweetid, retweets, links and much more.
### and the other informations about the csv file is explained in preprocess.py
