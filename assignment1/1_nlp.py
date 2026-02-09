import nltk
from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize import wordpunct_tokenize
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import MWETokenizer
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

wt = WhitespaceTokenizer()
tokens_ws = wt.tokenize(text)
print("Whitespace Tokenizer:", tokens_ws)


tokens_punct = wordpunct_tokenize(text)
print("Punctuation Tokenizer:", tokens_punct)

treebank = TreebankWordTokenizer()
tokens_treebank = treebank.tokenize(text)
print("Treebank Tokenizer:", tokens_treebank)

tweet = TweetTokenizer()
tokens_tweet = tweet.tokenize(text)
print("Tweet Tokenizer:", tokens_tweet)


mwe = MWETokenizer([('Machine', 'Learning')])
tokens_mwe = mwe.tokenize(text.split())
print("MWE Tokenizer:", tokens_mwe)
