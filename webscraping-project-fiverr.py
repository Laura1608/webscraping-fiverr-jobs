import numpy as np
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Create empty list to store the scraped content
data = []


for i in range(1, 10):
    url = 'https://nl.fiverr.com/search/gigs?query=web%20scraping&source=pagination&acmpl=1&search_in=everywhere&search-autocomplete-original-term=web%20scraping&search-autocomplete-available=true&search-autocomplete-type=suggest&search-autocomplete-position=1&ref_ctx_id=ea69547a50e642d59ebb396d06c3ea08&page={i}&offset=-1'

    # Create dictionary with user-agent = Firefox
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0', 'user_id': '151948336'}

    # Retrieve content from webpage
    response = requests.get(url, headers=headers)
    content = response.content
    soup = BeautifulSoup(content, 'html.parser')

    # Loop over every Fiverr gig to retrieve its details
    for services in soup.find_all('div', class_='basic-gig-card'):

        # Find the title element on the page
        title = services.find('h3', class_='QTdEgIS text-normal').text

        # Set standard to 'None' for rating, and replace with score if available
        rating = np.NaN
        gig_rating = services.find('b', class_='rating-score iLiXwIR').text
        if gig_rating:
            rating = gig_rating.replace(',', '.')
            rating = float(rating)

        # Set standard to '0' for review counts, and replace with value if available
        review_count = '0'
        reviews = services.find('span', class_='rating-count-number').text
        if reviews:
            review_count = int(reviews)

        # Set standard to '0' for level, and replace with value if available
        level = '0'
        level_class = services.find('div', class_='tbody-6 text-semi-bold claXVc8')
        if level_class:
            level = level_class.text.replace('Niveau ', '')
            level = int(level)

        # Find the price element on the page
        price = services.find('span', class_='text-bold co-grey-1200').span
        price = price.text.replace('€\xa0', '')
        price = int(price)

        # Add all variables to a list
        data.append([title, rating, review_count, level, price])


# Create a dataframe with columns
column_names = ['title', 'rating', 'review_count', 'level', 'price']
df = pd.DataFrame(data, columns=column_names)
print(df.head())

df.to_csv('webscraping-fiverr-output.csv', index=False)
print("Web Scraping Fiverr gigs extracted!")
