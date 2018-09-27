# tweet_scrape.py
Here i used selenium for scraping because selenium is a web automation tool so i can control my browser by python code and can scrape
![screenshot 2018-09-26 at 8 22 02 pm](https://user-images.githubusercontent.com/35501699/46089543-5fd3b800-c1cc-11e8-9d00-4a5e4886ce9c.png)
In the above screenshot i started by browser in my case i chose chrome(for this we need to install drivers) and then chose my url and then find 'body' element by according to tag_name and i scrolled the web page.
### Main Logic
![screenshot 2018-09-26 at 9 04 53 pm](https://user-images.githubusercontent.com/35501699/46091321-3583f980-c1d0-11e8-8e3e-3182f14f7799.png)
After the first part, extracted page source in html format and then parsed it using beautiful soup.
For userid of twitter account found element 'a' with class name as shown and i knew about the 'a' and class from html source(in chrome by inspecting element)
For tweet text element 'p' with class name as shown by the same way as stated above
### last part of storing values in word.txt file
![screenshot 2018-09-26 at 8 22 26 pm](https://user-images.githubusercontent.com/35501699/46092847-674a8f80-c1d3-11e8-84f5-045ad0cb2f3a.png)
    
In this part created a file name word.txt and for 200 iterations from user id part extracted attribute 'href' and for tweet extracted it's text part and wrote in word.txt file and i also included try and exception statement because if no. of tweets is less than 200 then it should not throw error.

### Note: regarding file.json
I also included file.json(and i generated file.json using twint python package) just for json_parse.py where i parsed this file.
