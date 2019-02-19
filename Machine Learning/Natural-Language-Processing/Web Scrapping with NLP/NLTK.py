import requests
from bs4 import BeautifulSoup

page = requests.get('https://www.gutenberg.org/files/2701/2701-h/2701-h.htm')

soup = BeautifulSoup(page.text, 'html5lib')
tags = soup.find_all('a')
'''
for tag in tags:
    print(tag.get_text('href'))
'''
text = soup.get_text()  
#print(text)

## Import RegexpTokenizer from nltk.tokenize
from nltk.tokenize import RegexpTokenizer

# Create tokenizer
tokenizer = RegexpTokenizer('\w+')

# Create tokens
tokens = tokenizer.tokenize(text)
#print(tokens[:8])

# Initialize an empty list
words = []

# Loop through list tokens and make lower case
for word in tokens:
    words.append(word.lower())

# Print several items from list as sanity check
#print(words[:8])

#import nltk
import nltk
#nltk.download('stopwords')

#get english stopwords and print some of them
sw = nltk.corpus.stopwords.words('english')
print(sw[:])

#initialize an empty list
words_ns = []

#add to words_ns all words that are in words list but not in sw list

for word in words:
    if word not in sw:
        words_ns.append(word)
        
        
#print the list items
print(words_ns[:9])
















