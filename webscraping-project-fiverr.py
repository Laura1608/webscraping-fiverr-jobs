import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd

# Create empty list to store the scraped content
data_list = []

# Create a loop to extract data from the first 10 pages
for i in range(1, 11):

    # Initialize Selenium WebDriver
    driver = webdriver.Firefox()

    # Create dictionary with user-agent = Firefox as a header
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0', 'user_id': '', 'locale': 'en-US'}

    # Connect to url through Selenium webdriver
    url = 'https://www.fiverr.com/search/gigs?query=scraping&source=pagination&ref_ctx_id=b7d6afc246514ab888c6631dfb5d6c7d&search_in=everywhere&search-autocomplete-original-term=scraping&page={i}&offset=-1'
    driver.get(url)

    # Before performing next action, wait two seconds
    time.sleep(2)

    # Select body element on webpage
    elem = driver.find_element('tag name', 'body')

    # Specify amount of scroll downs on page
    no_of_pagedowns = 4

    # Create loop to scroll down on page automatically to retrieve its data
    while no_of_pagedowns:
        elem.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.2)
        no_of_pagedowns -= 1

    # Fetch data using BeautifulSoup after all data is loaded
    soup = BeautifulSoup(driver.page_source, "html.parser")

    # Close the WebDriver session
    driver.quit()

    # Loop over all gigs to retrieve their details
    for gigs in soup.find_all('div', class_='basic-gig-card'):

        # Find the title element on the page
        title = gigs.find('h3', class_='QTdEgIS text-normal').text

        # Set standard to 'None' for rating, and replace with score if available
        # Change datatype to float
        rating = np.NaN
        gig_rating = gigs.find('b', class_='rating-score iLiXwIR')
        if gig_rating:
            rating = gig_rating.text.replace(',', '.')
            rating = float(rating)

        # Set standard to '0' for review counts, and replace with value if available
        review_count = '0 reviews'
        reviews = gigs.find('span', class_='rating-count-number')
        if reviews:
            review_count = reviews.text

        # Set standard to '0' for level, and replace with value if available
        # Change datatype to integer
        level = 'Level 0'
        level_class = gigs.find('div', class_='tbody-6 text-semi-bold claXVc8')
        if level_class:
            level = level_class.text

        # Set standard to 'No' for the Top Rated badge, and replace with value if available
        top_rated = 'No Top Rated badge'
        badge = gigs.find('span', class_='ucgAJ7j')
        if badge:
            top_rated_badge = badge.find('div', class_='tbody-6 text-semi-bold claXVc8 DK7KkjD')
            if top_rated_badge:
                top_rated = top_rated_badge.text

        # Set standard to 'No' for the Pro badge, and replace with value if available
        pro = 'No Pro badge'
        badge = gigs.find('span', class_='ucgAJ7j')
        if badge:
            pro_badge = badge.find('div', class_='g1R5_pQ')
            if pro_badge:
                pro = pro_badge.p.text

        # Set standard to 'No' for video consultation, and replace with value if available
        video_cons = 'No video consultation'
        consultation = gigs.find('p', class_='_1sz16f5k z58z871gm z58z871eo z58z877 z58z872')
        if consultation:
            video_cons = consultation.text

        # Set standard to 'No' for extra languages spoken, and replace with value if available
        language = 'No extra languages'
        seller_language = gigs.find('div', class_='seller-language')
        if seller_language:
            language = seller_language.span.text

        # Find the price element on the page
        # Change datatype to integer
        price = gigs.find('span', class_='text-bold co-grey-1200').span
        price = price.text.replace('€\xa0', '').replace('€', '')
        price = int(price)

        # Find the url of every gig
        url = gigs.find('a', class_='tbody-5 p-t-8 lt2ar2q EhHcMiw')['href']
        url = "https://nl.fiverr.com/" + url

        # Add all variables to a new list
        data_list.append([title, rating, review_count, level, top_rated, pro, video_cons, language, price, url])


# Create a dataframe with columns and print the first rows
column_names = ['title', 'rating', 'review_count', 'level', 'top_rated', 'pro_badge', 'video_cons', 'language', 'price', 'url']
df = pd.DataFrame(data_list, columns=column_names)

# Drop duplicated rows, based on url
df = df.drop_duplicates(subset='url')

df.to_csv('webscraping-fiverr-output.csv', sep=';', index=False)
