import json
import pandas as pd


data = []            # here json file is loaded and data appended in a list
with open('file.json') as f:
    for line in f:
        data.append(json.loads(line))


pan_data = pd.DataFrame([], columns=['S.No.', 'tweetid', 'tweettext'])


for ix in range(600):                # parsing a json format and accessing username and tweets
    tweetid = data[ix]['username']
    tweettext = data[ix]['tweet']
   
    app_data = {"S.No.":ix+1,
                "tweetid":tweetid,
                "tweettext":tweettext
               }
    
    pan_data = pan_data.append(app_data, ignore_index=True)


pd.DataFrame.to_csv(pan_data,'/Users/adityakumar/Desktop/tweets.csv', encoding='utf8')    # saving the data in csv format

