import pandas as pd
import plotly.express as px
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.probability import FreqDist
from nltk.util import ngrams
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Ignore certain warning in code
import warnings
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

# Start with data cleaning to prepare the data for the analysis
dataset = pd.read_csv('webscraping-fiverr-output.csv', delimiter=';')
data = pd.DataFrame(data=dataset)

# Column_names = ['title', 'rating', 'review_count', 'level', 'pro_badge', 'top_rated', 'video_cons', 'language', 'price', 'url']

# Cleaning text data columns and change their data type to integer
data['title'] = data['title'].replace('\n', ',')
data['review_count'] = data['review_count'].str.replace('k+', '000').str.replace(' reviews', '')
data['review_count'] = data['review_count'].astype(int)
data['level'] = data['level'].str.replace('Level ', '')
data['level'] = data['level'].astype(int)
data['language'] = data['language'].str.replace('I speak Spanish +1', '1').str.replace('I speak Spanish +2', '1').str.replace('I speak Spanish +3', '1').str.replace('No extra languages', '0')
data['language'] = data['language'].astype(int)
data['pro_badge'] = data['pro_badge'].str.replace('No Pro badge', '0').str.replace('Pro', '1')
data['pro_badge'] = data['pro_badge'].astype(int)
data['top_rated'] = data['top_rated'].str.replace('No Top Rated badge', '0').str.replace('Top Rated', '1')
data['top_rated'] = data['top_rated'].astype(int)
data['video_cons'] = data['video_cons'].str.replace('No video consultation', '0').str.replace('Offers video consultations', '1')
data['video_cons'] = data['video_cons'].astype(int)

data.info()

px.box(data['price']).show()
px.box(data['rating']).show()
px.box(data['review_count']).show()
px.box(data['level']).show()

# Check values in column
print(data['rating'].value_counts())
print(data['level'].value_counts())
print(data['top_rated'].value_counts())
print(data['pro_badge'].value_counts())
print(data['video_cons'].value_counts())
print(data['language'].value_counts())

# Remove columns 'pro_badge', because there isn't enough difference between values
data = data.drop(['pro_badge'], axis=1)

# Create new dataframe with only the variables we want to compare
data_new = data[['rating', 'review_count', 'level', 'top_rated', 'video_cons', 'language', 'price']]

# Calculate correlation coefficient for all combinations of variables
data_corr = data_new.corr().round(3)

# Show scores in heatmap for overview
px.imshow(data_corr).show()

# Prior knowledge:
# Correlation >=0.7 and <0.9 means there is a strong relationship between the variables.
# Correlation >=0.5 and <0.7 means there is a moderate relationship  between the variables.
# Correlation >=0.3 and <0.5 means there is a weak relationship between the variables.
# Correlation <0.3 means there is no relationship between the variables.

# FINDINGS:
# There is a weak relation (correlation coefficient: -0.44) between the variables 'level' and 'top_rated', which means they move in opposite directions (the higher the level, the less often a seller has the Top Rated badge).
# No further correlations found.


# Creating second dataset, to compare results of most successful gigs
# Define "successful" gigs, based on statistical distribution (> 75%):
print(data['review_count'].describe())
all_gigs = data.copy()
successful_gigs = data[(data['review_count'] > 230)]

# Print length of both datasets
print("All gigs amount: ", len(all_gigs), "\n" "Successful gigs amount: ", len(successful_gigs))


# Plot bar graphs of both datasets, to see their differences in pricing
px.bar(all_gigs['price'].value_counts(), title='Price occurrence in all gigs').show()
px.bar(successful_gigs['price'].value_counts(), title='Price occurrence in successful gigs').show()

# Plot bar graphs, to see the differences in level
px.bar(all_gigs['level'].value_counts(), title='Level occurrence in all gigs').show()
px.bar(successful_gigs['level'].value_counts(), title='Level occurrence in successful gigs').show()

# Plot bar graphs, to see the differences in 'Top Rated' badge
px.bar(all_gigs['top_rated'].value_counts(), title='Top Rated badge occurrence in all gigs').show()
px.bar(successful_gigs['top_rated'].value_counts(), title='Top Rated badge occurrence in successful gigs').show()

# Plot bar graphs, to see the differences in video consultation offer
px.bar(all_gigs['video_cons'].value_counts(), title='Video consultation occurrence in all gigs').show()
px.bar(successful_gigs['video_cons'].value_counts(), title='Video consultation occurrence in successful gigs').show()

# Plot bar graphs, to see the differences in language offer
px.bar(all_gigs['language'].value_counts(), title='Extra language occurrence in all gigs').show()
px.bar(successful_gigs['language'].value_counts(), title='Extra language occurrence in successful gigs').show()

# FINDINGS:
# - Successful gigs are more often at level 2 than all gigs.
# - The price for both groups is mostly at $30, followed by $20.
# - Having a 'Top Rated' badge is not a determinator for success. No difference in results.
# - Offering multiple languages on your profile doesn't seem to influence success, either.
# - Successful gigs offer video consultation more often.


# Group results by level, and calculate their average price
price_level_avg = all_gigs.groupby('level')['price'].mean()
price_level_avg_suc = successful_gigs.groupby('level')['price'].mean()

# Plot results in bar chart for better overview
px.bar(price_level_avg, labels={'value': 'price'}, title='Average price of all gigs, grouped by level').show()
px.bar(price_level_avg_suc, labels={'value': 'price'}, title='Average price of successful gigs, grouped by level').show()


# Group results by 'Top Rated' badge, and calculate their average price
price_toprated_avg = all_gigs.groupby('top_rated')['price'].mean()
price_toprated_avg_suc = successful_gigs.groupby('top_rated')['price'].mean()

# Plot results in bar chart for better overview
px.bar(price_toprated_avg, labels={'value': 'price'}, title='Average price of all gigs, grouped by Top Rated badge').show()
px.bar(price_toprated_avg_suc, labels={'value': 'price'}, title='Average price of successful gigs, grouped by Top Rated badge').show()


# Group results by offering video consultation, and calculate their average price
price_video_avg = all_gigs.groupby('video_cons')['price'].mean()
price_video_avg_suc = successful_gigs.groupby('video_cons')['price'].mean()

# Plot results in bar chart for better overview
px.bar(price_video_avg, labels={'value': 'price'}, title='Average price of all gigs, grouped by video consultation').show()
px.bar(price_video_avg_suc, labels={'value': 'price'}, title='Average price of successful gigs, grouped by video consultation').show()

# FINDINGS:
# - Sellers in level 0 ask for higher prices than sellers in level 1 or 2. This is especially true for successful gigs.
# - The average price in level 0 is higher for successful gigs, compared to all gigs ($45 vs $36).
# - Sellers with a Top Rated badge ask higher prices than sellers without the badge.
# - Sellers who offer video consultation ask higher prices than sellers who don't ($29 vs $34 for all gigs, and $27 vs $37 for successful gigs).


# Now start with the text analysis...
# Define stop word list
final_stopwords = nltk.corpus.stopwords.words('english')
final_stopwords.append('u')

# Create empty lists to save text later
all_gigs_all_titles = []
successful_gigs_all_titles = []


# Create function to pre-process text
def preprocess_text(row):
    # Tokenize every word in each row, remove interpunction, and change to lowercase
    text_title = RegexpTokenizer(r'\w+')
    tokens_title = text_title.tokenize(row['title'].lower())

    # Remove stop words
    filtered_tokens_title = [token for token in tokens_title if token not in final_stopwords]

    # Lemmatize the tokens for keeping base part only
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens_title = [lemmatizer.lemmatize(token) for token in filtered_tokens_title]

    # Join all tokens back into a string
    processed_text_title = ' '.join(lemmatized_tokens_title)

    # Calculate title length per row
    title_length = len(processed_text_title)

    # Append tokenized strings to empty lists for later analysis
    [all_gigs_all_titles.append(token) for token in lemmatized_tokens_title]
    [successful_gigs_all_titles.append(token) for token in lemmatized_tokens_title if data.loc[row.name, 'review_count'] > 230]

    return processed_text_title, title_length


# Apply function to dataframe, row by row (axis=1), and create new columns with output
all_gigs[['processed_title', 'title_length']] = all_gigs.apply(preprocess_text, axis=1, result_type='expand')
successful_gigs[['processed_title', 'title_length']] = successful_gigs.apply(preprocess_text, axis=1, result_type='expand')

# Calculate the average length of titles
print("Average title length of all gigs: ", all_gigs['title_length'].astype(int).mean().round(1))
print("Average title length of successful gigs: ", successful_gigs['title_length'].astype(int).mean().round(1))


# Final part of text analysis, focusing on word use in titles
# Frequency distribution to find most common words in titles of both datasets
print("Most common words in titles of all gigs: ", "\n", FreqDist(all_gigs_all_titles).most_common(5))
print("Most common words in titles of successful gigs: ", "\n", FreqDist(successful_gigs_all_titles).most_common(5))


# Pair words through ngrams and find most common 2-word combinations
all_gigs_bigram_title = list(ngrams(all_gigs_all_titles, 2))
all_gigs_common_bigram_title = FreqDist(all_gigs_bigram_title).most_common(5)
print("Most common 2-word combinations in titles of all gigs: ", "\n", all_gigs_common_bigram_title)

successful_gigs_bigram_title = list(ngrams(successful_gigs_all_titles, 2))
successful_gigs_common_bigram_title = FreqDist(successful_gigs_bigram_title).most_common(5)
print("Most common 2-word combinations in titles of successful gigs: ", "\n", successful_gigs_common_bigram_title)


# Pair words through ngrams and find most common 3-word combinations
all_gigs_trigram_title = list(ngrams(all_gigs_all_titles, 3))
all_gigs_common_trigram_title = FreqDist(all_gigs_trigram_title).most_common(5)
print("Most common 3-word combinations in titles of all gigs: ", "\n", all_gigs_common_trigram_title)

successful_gigs_trigram_title = list(ngrams(successful_gigs_all_titles, 3))
successful_gigs_common_trigram_title = FreqDist(successful_gigs_trigram_title).most_common(5)
print("Most common 3-word combinations in titles of successful gigs: ", "\n", successful_gigs_common_trigram_title)


# Create function to generate wordcloud
def plot_wordcloud(titles):
    # Concatenate all titles into a single string
    all_titles = ' '.join(titles)

    # Generate word cloud
    word_cloud = WordCloud(background_color="white")
    word_cloud.generate(all_titles)

    # Plot the word cloud
    plt.imshow(word_cloud, interpolation="bilinear")
    plt.axis('off')
    plt.title('Most used words in gig titles')
    plt.show()


# Apply function to list with titles
plot_wordcloud(all_gigs_all_titles)
plot_wordcloud(successful_gigs_all_titles)

# FINDINGS:
# - Successful gigs have the same title length as all gigs.
# - Both groups have the same words (or word combinations) used in titles.
# - Words that often occur in their titles: 'web scraping', 'scraping data', 'data mining', 'data entry', 'data scraping', 'python', and 'website'.


# Create function to perform sentiment analysis
def sentiment_analyzer(title):
    sentiment = TextBlob(title)
    polarity = sentiment.sentiment.polarity
    return polarity


# Apply function to dataframe, row by row (axis=1), and create new column with output
all_gigs['polarity'] = all_gigs['processed_title'].apply(sentiment_analyzer)
successful_gigs['polarity'] = successful_gigs['processed_title'].apply(sentiment_analyzer)

print("Polarity score of all gigs: ", all_gigs['polarity'].mean().round(3))
print("Polarity score of successful gigs: ", successful_gigs['polarity'].mean().round(3))


# Create function to perform subjectivity analysis
def subjectivity_analyzer(title):
    sentiment = TextBlob(title)
    subjectivity = sentiment.sentiment.subjectivity
    return subjectivity


# Apply function to dataframe, row by row (axis=1), and create new column with output
all_gigs['subjectivity'] = all_gigs['processed_title'].apply(subjectivity_analyzer)
successful_gigs['subjectivity'] = successful_gigs['processed_title'].apply(subjectivity_analyzer)

print("Subjectivity score of all gigs: ", all_gigs['subjectivity'].mean().round(3))
print("Subjectivity score of successful gigs: ", successful_gigs['subjectivity'].mean().round(3))

# Prior knowledge:
# Polarity scores are numerical values that range from -1 to 1, where -1 indicates a very negative sentiment, 0 indicates a neutral sentiment, and 1 indicates a very positive sentiment.
# Subjectivity scores are numerical values that range from 0 to 1, where 0 indicates a very objective text, and 1 indicates a very subjective text.

# FINDINGS:
# - Successful gigs have more neutrally and objectively written titles, compared to all gigs.


# --------------------------------------------

# OVERALL CONCLUSION:
# - The most common price in both groups is $30. However, successful sellers set their prices higher in level 0.
# - Having a 'Top Rated' or 'Pro' badge don't seem to influence success. Neither does offering multiple languages.
# - However, sellers ask for higher prices when having a 'Top Rated' badge or when offering video consultation.
# - Successful gigs offer video consultation more often. Also, they are more often in level 2.
# - The title length isn't of influence on the gig's success. Also, the same words are most used in both groups.
# - Successful gigs, however, use more neutrally and objectively written titles.

# Research questions:
# # Do successful gigs have a higher level on average? YES
# # Do successful gigs have a 'Pro' badge more often? NO
# # Do successful gigs have a 'Top Rated' badge more often? NO
# # Do successful gigs offer video consultation more often? YES
# # Do sellers ask a higher price when having a 'Top Rated' badge? YES
# # Do sellers ask a higher price when offering video consultation? YES
# # Do successful gigs ask higher prices on average? YES, ESPECIALLY IN LEVEL 0
# # What are text characteristics of successful title gigs? MORE NEUTRAL AND OBJECTIVE
