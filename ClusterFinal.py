
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
from pandas import *
from IPython.display import display, HTML
from sklearn.cluster import k_means
from sklearn.metrics import silhouette_score

get_ipython().magic(u'load_ext sql')
get_ipython().magic(u'sql sqlite:///followers.db')


# In[2]:

#Map languages to latitude/longitude
lines = [line.rstrip('\n') for line in open("languagemapping.txt")]
languageMap = {}
for line in lines:
    language = line.split(',')[0]
    latitude = line.split('[')[1].split(',')[0].strip()
    longitude = line.split('[')[1].split(',')[1].strip().strip(']')
    languageMap[language] = [float(latitude), float(longitude)]

print languageMap


# In[35]:

# Create training matrix X from sql tables
numSamples = 10000
numInfluencers = 12

X = [] # rows of sample data
rows = get_ipython().magic(u'sql SELECT * FROM Users LIMIT :numSamples;')

for row in rows:
    user = list(row)
    del(user[7]) #delete UtcOffset
    del(user[5]) #delete Verified
    del(user[0]) #delete ID 

    if user[1] == -1: #case we didn't finish filling out this user
        continue
    
    userID = user[0]
    influencers = get_ipython().magic(u'sql SELECT * FROM Influencers WHERE FollowerID = :userID ORDER BY InfluencerName;')

    if len(influencers) != numInfluencers:  #don't store incomplete data
        continue
    
    latitude = languageMap[user[4]][0] #Map string language to numerical value
    longitude = languageMap[user[4]][1]
    del(user[4])
    user.append(latitude)
    user.append(longitude)
    
    for j in range(0, numInfluencers):
        user.append(influencers[j][2])
    
    X.append(user[1:]) #ignore UserID field

X = np.array(X)
    
X_printable = DataFrame(X)
X_printable.columns = ['StatusesCount', 'FollowersCount', 'FollowingCount', 'Latitude', 'Longitude', 'BarackObama', 'BillGates', 'Cristiano', 'KimKardashian', 'LilTunechi', 'NASA', 'Oprah', 'Pontifex', 'cnnbrk', 'espn', 'jimmyfallon', 'taylorswift13']


# In[36]:

print X.shape
display(X_printable)


# In[67]:

display(X_printable[9810:9811])


# In[5]:

#Create histogram of influencers
#Nike data
names = get_ipython().magic(u'sql SELECT InfluencerName FROM Influencers GROUP BY InfluencerName;')
names = [str(name).split("\'")[1] for name in list(names)]
counts = np.sum(X, 0)[5:]
percents = counts/float(len(X))*100

#General population data
total_num_twitter_users = 320 * 10e5
Obama_count = 66.6 * 10e5
Gates_count = 25.8 * 10e5
Cristiano_count = 39.1 * 10e5
Kim_count = 37.4 * 10e5
Tunechi_count = 24.9 * 10e5
Nasa_count = 13.5 * 10e5
Oprah_count = 29.9 * 10e5
Pope_count = 8.03 * 10e5
CNN_count = 32 * 10e5
ESPN_count = 23.2 * 10e5
Fallon_count = 32.1 * 10e5
Swift_count = 66.9 * 10e5
num_total_followers = [Obama_count, Gates_count, Cristiano_count, Kim_count, Tunechi_count, Nasa_count, Oprah_count, Pope_count, CNN_count, ESPN_count, Fallon_count, Swift_count]
total_percents = [(x/total_num_twitter_users)*100 for x in num_total_followers]

#plot
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 14}

plt.rc('font', **font)
ax = plt.subplot(111)
N = len(percents)
ind = np.arange(N)  # the x locations for the groups
width = 0.35  
ind = np.arange(N) 
nike = ax.bar(ind,percents,width, color='mediumseagreen')
worldwide = ax.bar(ind+width, total_percents, width, color='y')

ax.set_xticks(ind+width)
ax.set_xticklabels(names,rotation=45, rotation_mode="anchor", ha="right")
ax.set_ylabel("% Following")
ax.legend( (nike, worldwide), ('Nike Followers Only', 'All Active Accounts'), loc='upper center' )

plt.title("Percentage of Followers by Influencer")
plt.tight_layout()
plt.savefig('influencers.png')
plt.show()


# In[5]:

#Dimensionality reduction on influencers
from sklearn import decomposition

#Extract just the features we want to do PCA on, ie the influencers
on_columns = range(5,17) 
X_PCA = X[:, on_columns] 

#Repeatedly run PCA on with varying dimensionality
explained_variances = []
for i in range (0,len(X_PCA[0])+1):
    pca = decomposition.PCA(n_components=i)
    pca.fit(X_PCA)
    explained_variances.append((pca.explained_variance_ratio_).sum())

#Plot explained_variances
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 14}

plt.rc('font', **font)

fig = plt.figure()
plt.subplot(111)
plt.plot(np.linspace(0, len(explained_variances)-1, len(explained_variances)), explained_variances,  linestyle='--', marker='o')
plt.title("Explained Variance by Dimensionality")
plt.ylabel("Explained Variance")
plt.xlabel("Dimensionality")

#Use lowest dimension that captures at least 85% of the variance
dim = [ n for n,i in enumerate(explained_variances) if i>0.85 ][0]
pca = decomposition.PCA(n_components=dim)
pca.fit(X_PCA)
X_PCA = pca.transform(X_PCA)

#replace the influencers in X with the new reduced-dimensionality columns in X_PCA
for i in reversed(range(5,17)):
    X = np.delete(X, i, 1) 
X = np.hstack((X, X_PCA))

plt.plot(dim,explained_variances[dim],'ro',fillstyle='none',markersize=30, mew=5)
plt.tight_layout()
plt.savefig('dimensionality.png')
plt.show()


# In[6]:

# Normalize dimension of [0:4] to have a similar range as that of [5:12] (which are the outputs of PCA)
Xdf = DataFrame(X)
a = Xdf[[5,6,7,8,9,10,11,12]].min().min()
b = Xdf[[5,6,7,8,9,10,11,12]].max().max()
Xdf[[0,1,2,3,4]] = (b - a)*(Xdf[[0,1,2,3,4]]/Xdf[[0,1,2,3,4]].max()) + a
X = np.array(Xdf)


# In[18]:

#Given k=5, just trying to figure out if who's in the clusters makes any sense
print X.shape
(centroids, labels, _) = k_means(X=X, n_clusters=5, random_state=170) 
i = 0
cluster0 = []
cluster1 = []
cluster2 = []
cluster3 = []
cluster4 = []
for l in labels:
    if (i%10 == 0):
        if(l == 0):
            cluster0.append(i)
        elif(l == 1):
            cluster1.append(i)
        elif(l == 2):
            cluster2.append(i)
        elif(l == 3):
            cluster3.append(i)
        elif(l == 4):
            cluster4.append(i)
    i+=1


# In[34]:

#ctd from above
print cluster4


# In[9]:

#Find the ideal number of clusters
import math 

#Repeatedly run k-means with different cluster values
dimension = len(X[0])
scores = [-1, -1]#disallow clusters of size 0 or 1
for k in range (2,15): #upper limit on number of clusters
    (centroids, labels, _) = k_means(X=X, n_clusters=k, random_state=170)    
    score = silhouette_score(X=X, labels=labels)
    scores.append(score)

candidates = np.where(np.array(scores) >= .7)[0] # Take first cluster size greater than a threshold
if len(candidates) > 0:
    k_hat = candidates[0]
else:
    k_hat = np.where(np.array(scores).max() == scores)[0][0] # Take maximum

print "k_hat = " + str(k_hat) + " with score = " + str(scores[k_hat])


# In[125]:

#Plot error 
fig = plt.figure()
plt.subplot(111)
plt.plot(np.linspace(0, len(scores)-1, len(scores)), scores, linestyle='--', marker='o')
plt.plot(k_hat,scores[k_hat],'ro',fillstyle='none',markersize=20, mew=3)
plt.xlabel('k')
plt.ylabel('Silhouette Coefficient')
plt.title('Silhouette Coefficient vs. k')
plt.savefig('silhouette_coefficient.png')
plt.show()


# In[126]:

def computeqscore(X, k):
    (centroids, labels, _) = k_means(X=X, n_clusters=k, random_state=170)
    X = np.array(X)
    centroids = np.array(centroids)
    #compute intra-distance
    intra_cluster_distances = []
    for i in range(0,len(centroids)):
        members = X[np.where(labels == i)]
        member_distances = [np.linalg.norm(centroids[i]-member) for member in members]
        intra_cluster_distances.append(np.mean(member_distances))

    #Compute inter-cluster distances
    inter_cluster_distances = []
    for i in range(0,len(centroids)):
        total_distance = 0
        for j in range(0,len(centroids)):
            total_distance += np.linalg.norm(np.array(centroids[i])-np.array(centroids[j]))
        inter_cluster_distances.append(np.mean(np.array(total_distance)))

    return sum(intra_cluster_distances)/sum(inter_cluster_distances)


# In[127]:

print computeqscore(X, k_hat)


# In[ ]:

#Plot clusters in R2 with same diameter/distance ratio
#-----------------------------------------------------
import random
k = 5
ratio = computeqscore(X, k_hat)#.04575
#randomly pick centroids
centroids = []
for i in range(0,k):
    centroids.append([random.random(), random.random()])
x = [row[0] for row in centroids]
y = [row[1] for row in centroids]

#compute mean inter-cluster distance
centroids = np.array(centroids)
mean_inter_distance = 0
total_distance = 0
for i in range(0,len(centroids)):
    for j in range(0,len(centroids)):
        total_distance += np.linalg.norm(centroids[i]-centroids[j])
        
mean_inter_distance = float(total_distance)/(len(centroids) ** 2)

#compute what mean intra-cluster distance should be
mean_intra_distance = mean_inter_distance*ratio

#generate list of intra_cluster distances with correct mean
radii = [mean_intra_distance for i in range(0, len(centroids))]
#make the circles slightly different sizes
for i in range(0,20):
    index1 = math.floor(random.random()*len(centroids))
    index2 = math.floor(random.random()*len(centroids))
    value = radii[int(index1)]/3
    radii[int(index1)] = radii[int(index1)] - value
    radii[int(index2)] = radii[int(index2)] + value
#red, blue, darkviolet, orange
colors = ['#FF0000', '#0000FF', '#9400D3', '#FF8C00', '#008000']
for i in range(0, k):
    circle = plt.Circle(centroids[i], radius=radii[i], alpha=0.6, color=colors[i%len(colors)])
    plt.gca().add_patch(circle)
plt.scatter([row[0] for row in centroids], [row[1] for row in centroids])
plt.axis("equal")
plt.title("Representative clusters with q = " + str(ratio))
plt.savefig("representativeclusters.png")
plt.show()



# In[ ]:

#Plot quality histogram
q_hat = .04575
k = 5
n_datapoints = 100

#generate random data
R = [[random.random(), random.random()] for i in range(0,n_datapoints)]

#generate strongly clustered "perfect" data around centroids
P = []
centroids = [[random.random(), random.random()] for i in range(0,k)]
for centroid in centroids:
    P.append(centroid)
    for i in range(0,int(math.ceil(n_datapoints/k))):
        P.append([centroid[0]+(random.random()/50),centroid[1]+(random.random()/50)])

#compute qscores
scores = [computeqscore(R,k), q_hat, computeqscore(P,k)]

fig = plt.figure()
ax1 = plt.subplot2grid((3,4), (0,0), colspan=2)
ax1.scatter([row[0] for row in R], [row[1] for row in R], c='r')
ax1.set_yticklabels('',visible=False)
ax1.set_xticklabels('',visible=False)
ax1.set_title("Random Data")

ax2 = plt.subplot2grid((3,4), (0,2), colspan=2)
ax2.scatter([row[0] for row in P], [row[1] for row in P])
ax2.set_yticklabels('',visible=False)
ax2.set_xticklabels('',visible=False)
ax2.set_title("Strongly-clustered \"Perfect\" Data")

ax3 = plt.subplot2grid((3,4), (1, 1), rowspan = 2, colspan = 2)
ind = np.arange(100)  # the x locations for the groups
width = 0.35  
ax.set_xticks(ind)
names = ['',"Random",'', "Ours",'', "Perfect"]
ax3.set_xticklabels(names,rotation=0, rotation_mode="anchor",)
index = np.arange(3)
ax3.bar(index,scores, width, color = ['r', 'g', 'b'], align="center")
ax3.set_title("Quality Metric Scores")
ax.set_xticks(index[:-1])
fig.tight_layout()
plt.savefig("qualitymetricscores.png")
plt.show()




# In[ ]:




# In[ ]:




# In[ ]:



