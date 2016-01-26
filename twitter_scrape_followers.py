
# coding: utf-8

# In[ ]:

get_ipython().magic(u'load_ext sql')
get_ipython().magic(u'sql sqlite:///followers.db')


# In[ ]:

get_ipython().run_cell_magic(u'sql', u'', u'CREATE TABLE IF NOT EXISTS Users (ID INTEGER PRIMARY KEY AUTOINCREMENT, UserID INT UNIQUE NOT NULL, StatusesCount INT NOT NULL, FollowersCount INT NOT NULL, FollowingCount INT NOT NULL, Verified INT NOT NULL, Language TEXT NOT NULL, UtcOffset INT);\nCREATE TABLE IF NOT EXISTS Influencers (FollowerID INT, InfluencerName TEXT, Following INT, PRIMARY KEY(FollowerID, InfluencerName), FOREIGN KEY(FollowerID) REFERENCES Users(UserID));')


# In[3]:

import tweepy
import time

auth = tweepy.OAuthHandler(consumer_key='lzrW1cKCWiQByFkXMU29Z7g6b',
                          consumer_secret='Hh6gGSb3YI3QiAeAU2NwWY4CvQdVstv7R8MXTufzejh1RtlMYS')
auth.set_access_token('4263905652-jXSdcPLRGXeuvznK56hnKkL0Lna0xR7AEX2NZuZ',
                      'Kpacok4p2Alr1oPki8oSOCOsytpFV8kZzUNar6GW8N6rf')
api = tweepy.API(auth, wait_on_rate_limit=True)


# In[ ]:

## Gather follower IDs for company
company = '@Nike'
pages = []
for page in tweepy.Cursor(api.followers_ids, id=company).pages(20):
    pages.extend(page)


# In[ ]:

## Insert follower IDs into Users table
for userID in pages:
    try:
        statuses_count = -1
        followers_count = -1
        following_count = -1
        verified = -1
        language = ''
        utc_offset = -1
        get_ipython().magic(u'sql INSERT INTO Users (UserID, StatusesCount, FollowersCount, FollowingCount, Verified, Language, UtcOffset) VALUES (:userID, :statuses_count, :followers_count, :following_count, :verified, :language, :utc_offset);')
    except:
        continue


# In[ ]:

## Look up follower information for each follower
kNumUsersPerRequest = 99
followerIDs = get_ipython().magic(u'sql SELECT UserID FROM Users WHERE ID <= 10000 AND Verified == -1;')
followerIDs = [followerID[0] for followerID in followerIDs]
i = 0
while i < len(followerIDs):
    try:
        followers = api.lookup_users(user_ids=followerIDs[i : (i + min(kNumUsersPerRequest, len(followerIDs) - i))])
        print time.strftime("*** %m/%d %H:%M:%S"), 'users =', i, 'through', (i + min(kNumUsersPerRequest, len(followerIDs) - i))
        i += kNumUsersPerRequest
    except tweepy.TweepError:
        i += kNumUsersPerRequest
        continue
                
    for follower in followers:
        followerID = follower.id
        print time.strftime("%m/%d %H:%M:%S"), 'followerID =', followerID

        statuses_count = follower.statuses_count
        followers_count = follower.followers_count
        following_count = follower.friends_count
        verified = follower.verified
        language = follower.lang
        utc_offset = follower.utc_offset
        get_ipython().magic(u'sql UPDATE Users SET StatusesCount = :statuses_count, FollowersCount = :followers_count, FollowingCount = :following_count, Verified = :verified, Language = :language, UtcOffset = :utc_offset WHERE UserID == :followerID;')


# In[ ]:



