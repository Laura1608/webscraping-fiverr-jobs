import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd

# Set page number count from 1
# i = 1
#
# while True:
#
# for i in range(1, 20):

# Initialize WebDriver
driver = webdriver.Firefox()

# Create dictionary with user-agent = Firefox
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0', 'user_id': '', 'locale': 'en-US'}

url = 'https://www.fiverr.com/search/gigs?query=scraping&source=pagination&ref_ctx_id=b7d6afc246514ab888c6631dfb5d6c7d&search_in=everywhere&search-autocomplete-original-term=scraping&page=2&offset=-1'
driver.get(url)

time.sleep(1)

elem = driver.find_element('tag name', 'body')

no_of_pagedowns = 4

while no_of_pagedowns:
    elem.send_keys(Keys.PAGE_DOWN)
    time.sleep(0.2)
    no_of_pagedowns -= 1

# Fetch the data using BeautifulSoup after all data is loaded
soup = BeautifulSoup(driver.page_source, "html.parser")

# Close the WebDriver session
driver.quit()

# Create empty list to store the scraped content
data_list = []

# Loop over every Fiverr gig to retrieve its details
for gigs in soup.find_all('div', class_='basic-gig-card'):

    # Find the title element on the page
    title = gigs.find('h3', class_='QTdEgIS text-normal').text

    # Set standard to 'None' for rating, and replace with score if available
    # Change datatype to float
    rating = np.NaN
    gig_rating = gigs.find('b', class_='rating-score iLiXwIR').text
    if gig_rating:
        rating = gig_rating.replace(',', '.')
        rating = float(rating)

    # Set standard to '0' for review counts, and replace with value if available
    review_count = '0 reviews'
    reviews = gigs.find('span', class_='rating-count-number').text
    if reviews:
        review_count = reviews

    # Set standard to '0' for level, replace with value if available
    # Change datatype to integer
    level = 'Level 0'
    level_class = gigs.find('div', class_='tbody-6 text-semi-bold claXVc8')
    if level_class:
        level = level_class.text

    pro_badge = 'No Pro badge'
    pro = gigs.find('span', class_='ucgAJ7j')
    if pro:
        pro_badge = "Pro badge"

    # Find out if video consultation is offered
    video_cons = 'No video consultation'
    consultation = gigs.find('p', class_='_1sz16f5k z58z871gm z58z871eo z58z877 z58z872')
    if consultation:
        video_cons = consultation.text

    language = 'No extra languages'
    seller_language = gigs.find('div', class_='seller-language')
    if seller_language:
        language = seller_language.span

    # Find the price element on the page
    # Change datatype to integer
    price = gigs.find('span', class_='text-bold co-grey-1200').span
    price = price.text.replace('€\xa0', '').replace('€', '')
    price = int(price)

    url = gigs.find('a', class_='tbody-5 p-t-8 lt2ar2q EhHcMiw')['href']
    url = "https://nl.fiverr.com/" + url

    # Add all variables to a list
    data_list.append([title, rating, review_count, level, pro_badge, video_cons, language, price, url])


# # Add count to i, to find the next page
# i = i + 1
# if not i:
#     break

# Create a dataframe with columns and print the first rows
column_names = ['title', 'rating', 'review_count', 'level', 'pro_badge', 'video_cons', 'language', 'price', 'url']
df = pd.DataFrame(data_list, columns=column_names)

df = df.drop_duplicates(subset='url')

print(df['pro_badge'].value_counts())
print(df['language'].value_counts())

print("\n Web Scraping Fiverr gigs extracted!")
