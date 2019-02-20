import re
import seaborn as sns
import requests
from bs4 import BeautifulSoup as bs
from nltk.tokenize import RegexpTokenizer 

#url of the page 
url = 'https://www.gutenberg.org/files/58091/58091-h/58091-h.htm'
#get the page request
page = requests.get(url)

soup = bs(page.text, 'html.parser')

texts = soup.find_all('p')
text = soup.get_text()
#print(text)
    
# Create tokenizer
tokenizer = RegexpTokenizer('\w+')
# Create tokens
tokens = tokenizer.tokenize(text)
# Initialize new list
#print(tokens)

words = []

for word in tokens:
    words.append(word.lower())
    
#print(words)

# Get English stopwords and print some of them
sw = nltk.corpus.stopwords.words('english')
# Initialize new list
#print(sw)

words_ns = []

for word in words:
    if word not in sw:
        words_ns.append(word)
        
#print(words_ns)

# Create freq dist and plot
freqdist1 = nltk.FreqDist(words_ns)
print(freqdist1.plot(20))



