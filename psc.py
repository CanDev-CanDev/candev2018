import pandas as pd
import numpy as np
from textblob import TextBlob
#from textblob.sentiments import NaiveBayesAnalyzer
import matplotlib.pyplot as plt

# needed for NaiveBayesAnalyzer
#import nltk
#nltk.download('movie_reviews')

data = pd.read_csv('PSC_Training_Dataset.csv')

data['polarity'] = data.apply(lambda x: TextBlob(x['answer']).sentiment.polarity, axis=1)
data['subjectivity'] = data.apply(lambda x: TextBlob(x['answer']).sentiment.subjectivity, axis=1)

#Sentiments = []
#for row in data['answer']:
#    Sentiments.append(TextBlob(row).sentiment)



# assign labels based on polarity
counts = np.zeros(5)
for i, polarity in enumerate(data['polarity']):
    if polarity >= 0.6:
        data.at[i, 'sentiment'] = 'vpos'
        counts[4] += 1
    elif polarity < 0.6 and polarity >= 0.2:
        data.at[i, 'sentiment'] = 'pos'
        counts[3] += 1
    elif polarity < -0.6:
        data.at[i, 'sentiment'] = 'vneg'
        counts[0] += 1
    elif polarity > -0.6 and polarity <= -0.2:
        data.at[i, 'sentiment'] = 'neg'
        counts[1] += 1
    else:
        data.at[i, 'sentiment'] = 'neu'
        counts[2] += 1

fig, ax = plt.subplots()

ax.set_ylim(0, 1000)
ax.set_xlim(0, 5)
# plot the data points
ax.bar(['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive'], counts)

# also plot last curve
#ax.plot(x, np.polyval(np.flip(weights[-1]), x), label=str(labels[-1]))

ax.set_xlabel('Polarity')
ax.set_ylabel('Number of Answers')

#title = "Degree {} polynomial with {} data points generated with variance {}".format(degree-1, len(D), sigma)
title = 'Number of Answers in Each Cateogry'
ax.set_title(title)

#plt.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
#lgd = plt.legend(loc='lower left', bbox_to_anchor=(
#    1., 0.0), ncol=2, borderaxespad=0, frameon=False)

#plt.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)
filename = "plot_{}.png".format(title)
plt.savefig(filename)
# plt.show()
plt.close()  # close the image

# Bayes runs too slow!
# data['prob+'] = data.apply(lambda x: TextBlob(x['answer'], analyzer=NaiveBayesAnalyzer()).sentiment, axis=1)
# data['prob1'] = data.apply(lambda x: TextBlob(x['answer'], analyzer=NaiveBayesAnalyzer())., axis=1)
#print("Number of very positives: {}".format(len(data[data['sentiment'] == 'vpos'])))
#print("Number of positives: {}".format(len(data[data['sentiment'] == 'pos'])))
#print("Number of neutrals: {}".format(len(data[data['sentiment'] == 'neu'])))
#print("Number of negatives: {}".format(len(data[data['sentiment'] == 'neg'])))
#print("Number of very negatives: {}".format(len(data[data['sentiment'] == 'vneg'])))

pd.DataFrame.to_excel(data, 'results.xls')



# print("Bad")
# print()
# print("\nGood")
# print(VeryGood)

# print(data)```
