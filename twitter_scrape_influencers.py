
# coding: utf-8

# In[ ]:

get_ipython().magic(u'load_ext sql')
get_ipython().magic(u'sql sqlite:///followers.db')


# In[53]:

import tweepy
import time

auth = tweepy.OAuthHandler(consumer_key='lzrW1cKCWiQByFkXMU29Z7g6b',
                           consumer_secret='Hh6gGSb3YI3QiAeAU2NwWY4CvQdVstv7R8MXTufzejh1RtlMYS')
#auth = tweepy.AppAuthHandler(consumer_key='lzrW1cKCWiQByFkXMU29Z7g6b',
#                             consumer_secret='Hh6gGSb3YI3QiAeAU2NwWY4CvQdVstv7R8MXTufzejh1RtlMYS')
auth.set_access_token('4263905652-jXSdcPLRGXeuvznK56hnKkL0Lna0xR7AEX2NZuZ',
                      'Kpacok4p2Alr1oPki8oSOCOsytpFV8kZzUNar6GW8N6rf')
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)


# In[63]:

## Look up influencer relationships for each follower
#influencers = ['@taylorswift13', '@espn', '@BillGates', '@Pontifex', '@BarackObama', '@KimKardashian', '@cnnbrk', '@jimmyfallon', '@Cristiano', '@LilTunechi', '@NASA', @Oprah']
influencer = '@Oprah'
followers = get_ipython().magic(u'sql SELECT UserID FROM Users WHERE ID <= 10000 AND Verified <> -1 AND UserID NOT IN (SELECT FollowerID FROM Influencers WHERE InfluencerName == :influencer);')
for follower in followers:    
    followerID = follower[0]
    
    # Check for influencers
    try:
        following = api.show_friendship(source_id=followerID, target_screen_name=influencer)[0].following
    except tweepy.TweepError as e:
        print e
        continue
        
    print time.strftime("%m/%d %H:%M:%S"), 'followerID =', followerID
        
    try:
        get_ipython().magic(u'sql INSERT INTO Influencers VALUES (:followerID, :influencer, :following);')
    except:
        continue


# In[ ]:



