import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.probability import FreqDist
from nltk.util import ngrams
from textblob import TextBlob
from wordcloud import WordCloud

# Ignore certain warning in code
import warnings
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

# Start with data cleaning to prepare the data for the analysis
dataset = pd.read_csv('webscraping-fiverr-output.csv', delimiter=';')
data = pd.DataFrame(data=dataset)

print("Variables: title, rating, review_count, level, pro_badge, top_rated_badge, video_consultation, extra_language, price, url")

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

# data.info()

# px.box(data['price']).show()
# px.box(data['rating']).show()
# px.box(data['review_count']).show()
# px.box(data['level']).show()

# Create new dataframe with only the variables we want to compare
data_new = data[['rating', 'review_count', 'level', 'top_rated', 'pro_badge', 'video_cons', 'language', 'price']]

# Calculate correlation coefficient for all combinations of variables
data_corr = data_new.corr().round(3)

# Show scores in heatmap for overview
px.imshow(data_corr, title='Correlation matrix variables').show()

print("Prior knowledge:")
print("Correlation >=0.7 and <0.9 (or negative) means there is a strong relationship between the variables.")
print("Correlation >=0.5 and <0.7 (or negative) means there is a moderate relationship.")
print("Correlation >=0.3 and <0.5 (or negative) means there is a weak relationship.")
print("Correlation <0.3 (or negative) means there is no relationship between the variables.")


# Creating second dataset, to compare results of most and less successful gigs
# Define "successful" and "non-successful" gigs, based on statistical distribution (<25% and >75%):
data['review_count'].describe()
all_gigs = data.copy()
successful_gigs = data[(data['review_count'] > 230)]
non_successful_gigs = data[(data['review_count'] > 27)]

# Print length of both datasets
print("All gigs amount: ", len(all_gigs), "\n" "Successful gigs amount: ", len(successful_gigs), "\n" "Non-successful gigs amount: ", len(non_successful_gigs))


# Count price occurrences in dataset, and sort them in ascending order
# price_vc_suc = successful_gigs['price'].value_counts().head()
# price_vc_non_suc = non_successful_gigs['price'].value_counts().head()

# Plot bar graphs of both datasets, to see their differences in pricing
price_vc_suc = successful_gigs['price'].value_counts().head()
px.pie(names=price_vc_suc.index, values=price_vc_suc.values, title='Price occurrence in successful gigs').update_traces(textinfo='percent').update_layout(legend_title_text='price in $').show()

price_vc_non_suc = non_successful_gigs['price'].value_counts().head()
px.pie(names=price_vc_non_suc.index, values=price_vc_non_suc.values, title='Price occurrence in non-successful gigs').update_traces(textinfo='percent').update_layout(legend_title_text='price in $').show()


# Group results by level, and calculate their average price
price_level_avg_suc = successful_gigs.groupby('level')['price'].mean()
price_level_avg_non_suc = non_successful_gigs.groupby('level')['price'].mean().head()

# Plot results in bar chart for better overview
px.bar(price_level_avg_suc, labels={'value': 'price'}, title='Average price of successful gigs, grouped by level').update_xaxes(dtick=1).show()
px.bar(price_level_avg_non_suc, labels={'value': 'price'}, title='Average price of non-successful gigs, grouped by level').update_xaxes(dtick=1).show()

# Creating dataframe with the value counts, as a preparation for plotting
total_vc_suc = successful_gigs[['level', 'pro_badge', 'top_rated', 'language', 'video_cons']].apply(pd.Series.value_counts)
total_vc_suc = total_vc_suc.transpose()
# Plot bar chart with the value counts per column
px.bar(total_vc_suc, orientation='h', labels={'value': 'gigs', 'index': 'variable', 'variable': 'value'}, title='Variation of variables occurring in successful gigs dataset').show()


# Creating dataframe with the value counts, as a preparation for plotting
total_vc_non_suc = non_successful_gigs[['level', 'pro_badge', 'top_rated', 'language', 'video_cons']].apply(pd.Series.value_counts)
total_vc_non_suc = total_vc_non_suc.transpose()
# Plot bar chart with the value counts per column
px.bar(total_vc_non_suc, orientation='h', labels={'value': 'gigs', 'index': 'variable', 'variable': 'value'}, title='Variation of variables occurring in non-successful gigs dataset').show()


# Now start with the text analysis...
# Define stop word list
final_stopwords = nltk.corpus.stopwords.words('english')
final_stopwords.append('u')

# Create empty lists to save text later
successful_gigs_all_titles = []
non_successful_gigs_all_titles = []


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
    [successful_gigs_all_titles.append(token) for token in lemmatized_tokens_title if data.loc[row.name, 'review_count'] > 230]
    [non_successful_gigs_all_titles.append(token) for token in lemmatized_tokens_title if data.loc[row.name, 'review_count'] > 27]

    return processed_text_title, title_length


# Apply function to dataframe, row by row (axis=1), and create new columns with output
successful_gigs[['processed_title', 'title_length']] = successful_gigs.apply(preprocess_text, axis=1, result_type='expand')
non_successful_gigs[['processed_title', 'title_length']] = non_successful_gigs.apply(preprocess_text, axis=1, result_type='expand')


# Calculate the average length of titles
print("Average title length of successful gigs: ", successful_gigs['title_length'].astype(int).mean().round(1), "characters")
print("Average title length of non-successful gigs: ", non_successful_gigs['title_length'].astype(int).mean().round(1), "characters")


successful_gigs_bigram_title = list(ngrams(successful_gigs_all_titles, 2))
successful_gigs_common_bigram_title = FreqDist(successful_gigs_bigram_title).most_common(5)
print("Most common 2-word combinations in titles of successful gigs: ", "\n", successful_gigs_common_bigram_title)

non_successful_gigs_bigram_title = list(ngrams(non_successful_gigs_all_titles, 2))
non_successful_gigs_common_bigram_title = FreqDist(non_successful_gigs_bigram_title).most_common(5)
print("Most common 2-word combinations in titles of non-successful gigs: ", "\n", non_successful_gigs_common_bigram_title)


# Create function to generate wordcloud
def plot_wordcloud(titles, title):
    # Concatenate all titles into a single string
    all_titles = ' '.join(titles)

    # Generate word cloud
    wordcloud = WordCloud(background_color="white")
    wordcloud.generate(all_titles)

    # Plot the word cloud
    plt.imshow(wordcloud)
    plt.title(title)
    plt.axis('off')
    plt.show()


# Apply function to list with titles
plot_wordcloud(successful_gigs_all_titles, 'Most used words in successful gig titles \n')
plot_wordcloud(non_successful_gigs_all_titles, 'Most used words in non-successful gig titles \n')


# Create function to perform sentiment analysis
def sentiment_analyzer(title):
    sentiment = TextBlob(title)
    polarity = sentiment.sentiment.polarity
    return polarity


# Apply function to dataframe, row by row (axis=1), and create new column with output
successful_gigs['polarity'] = successful_gigs['processed_title'].apply(sentiment_analyzer)
non_successful_gigs['polarity'] = non_successful_gigs['processed_title'].apply(sentiment_analyzer)

print("Polarity score of successful gig titles: ", successful_gigs['polarity'].mean().round(3))
print("Polarity score of non-successful gig titles: ", non_successful_gigs['polarity'].mean().round(3))


# Create function to perform subjectivity analysis
def subjectivity_analyzer(title):
    sentiment = TextBlob(title)
    subjectivity = sentiment.sentiment.subjectivity
    return subjectivity


# Apply function to dataframe, row by row (axis=1), and create new column with output
successful_gigs['subjectivity'] = successful_gigs['processed_title'].apply(subjectivity_analyzer)
non_successful_gigs['subjectivity'] = non_successful_gigs['processed_title'].apply(subjectivity_analyzer)

print("Subjectivity score of successful gig titles: ", successful_gigs['subjectivity'].mean().round(3))
print("Subjectivity score of non-successful gig titles: ", non_successful_gigs['subjectivity'].mean().round(3))

print("Prior knowledge:")
print("Polarity scores are numerical values that range from -1 to 1, where -1 indicates a very negative sentiment, 0 indicates a neutral sentiment, and 1 indicates a very positive sentiment.")
print("Subjectivity scores are numerical values that range from 0 to 1, where 0 indicates a very objective text, and 1 indicates a very subjective text.")


# Create sub-dataset with most successful gigs and their characteristics
successful_gigs_final = data[(data['review_count'] > 230) & (data['rating'] > 4.8) & (data['video_cons'] == 1) & (data['level'] == 2)]
unique_gigs_final = successful_gigs_final.drop_duplicates(subset=['title', 'review_count', 'level', 'price'])
print(len(unique_gigs_final))

# Print urls to look at the profiles in more detail
print("Gigs with successful characteristics, to take as an example for optimizing packages and profile text:")
for url in unique_gigs_final["url"]:
    print(url, "\n")
