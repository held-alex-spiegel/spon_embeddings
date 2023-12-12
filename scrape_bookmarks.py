import os
from bs4 import BeautifulSoup

# Directory path containing HTML files
folder_path = './data/my_bookmarks/'

# List to store article IDs from all files
all_article_ids = []

# Iterate through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.html'):
        file_path = os.path.join(folder_path, filename)
        # Read the HTML file
        with open(file_path, 'r', encoding='utf-8') as file:
            html = file.read()
        soup = BeautifulSoup(html, 'html.parser')
        # Find all elements with data-sara-article-id
        elements = soup.find_all('article', attrs={"data-sara-article-id": True})
        # Extract and save data-sara-article-id values
        article_ids = [element['data-sara-article-id'] for element in elements]
        # Add the article IDs to the list
        all_article_ids.extend(article_ids)

# Print the list of all article IDs
for article_id in all_article_ids:
    print(article_id)
    
# save to file
with open(folder_path+'article_ids.txt', 'w') as f:
    for item in all_article_ids:
        f.write("%s\n" % item)

