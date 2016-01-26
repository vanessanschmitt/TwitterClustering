
# coding: utf-8

# In[3]:

get_ipython().magic(u'load_ext sql')
get_ipython().magic(u'sql sqlite:///followers.db')


# In[4]:

influencer = '@NASA'
filename = 'influencer_' + influencer.replace('@','') +'.txt'

with open(filename,'r') as f_in:
    for line in f_in: 
        line = line.rstrip()
        pieces = line.split(' ')
        id = pieces[3].split('=')[1]
        following = int(pieces[4].split('=')[1] == 'True')
        print id,following
        try: 
            get_ipython().magic(u'sql INSERT INTO Influencers (FollowerID, InfluencerName, Following) VALUES (:id, :influencer, :following);')
        except:
            print 'failed'
            continue


# In[ ]:



