import re

# Define sentence
sentence = 'peter piper pick a peck of pickled peppers '

#define regex
ps = 'p\w+'

# Find all words in sentence that match the regex and print them
print(re.findall(ps, sentence))