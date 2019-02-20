import requests
from bs4 import BeautifulSoup
import re

page = requests.get('https://www.gutenberg.org/files/2701/2701-h/2701-h.htm')

soup = BeautifulSoup(page.text, 'html5lib')
tags = soup.find_all('a')
'''
for tag in tags:
    print(tag.get_text('href'))
'''
text = soup.get_text()  
print(text)

#finding the word starting with 'mr' in the text object
tokens = re.findall('mr\w+',text)
print(tokens)