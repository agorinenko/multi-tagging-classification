"""
Разбивает текст на токены и строит словарь
"""
import string
from typing import List
import pandas as pd
import nltk
import ssl
import pickle

from bs4 import BeautifulSoup
from torchtext.vocab import build_vocab_from_iterator

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
    
        
nltk.download('stopwords')
nltk.download('punkt')


def parse_tags(value):
        tags = value.replace('<', '').split('>')
        return [tag for tag in tags if tag]
    
def preprocessor(text):
    if not isinstance(text, str):
          return text
      
    tokenizer = TextPreProcessor(tokenizer=NltkTokenizer())    
    words = tokenizer.tokenize(text)
    return words
    
def yield_tokens(dataset):
    for tokens in dataset['Tokenize_title']:
        yield tokens
        
    for tokens in dataset['Tokenize_body']:
        yield tokens
        
stop_words = set(stopwords.words('russian'))
punctuation = set(string.punctuation)


class TextPreProcessor:
    def __init__(self, tokenizer, stemmer=None, morph=None):
        self.tokenizer = tokenizer
        self.stemmer = stemmer
        self.morph = morph


    def tokenize(self, text: str)->List[str]:    
        text = text.lower()
      
        doc = BeautifulSoup(text, 'lxml')
        text = doc.text
        
        tokens = self.tokenizer.tokenize(text)
        
        words = [word for word in tokens if word not in stop_words and word not in punctuation]
        
        if self.morph:
            words = [self.morph.parse(word)[0].normal_form for word in words]

        if self.stemmer:
            words = [self.stemmer.stem(word) for word in words]

        return words
    
class NltkTokenizer:    
    def tokenize(self, text: str):      
        return list(word_tokenize(text))
    
def save_vocab(vocab, path):
    with open(path, 'wb') as output:
        pickle.dump(vocab, output)

if __name__ == '__main__':
    file_path = 'data/stackoverflow_posts.csv'
    output_path = 'data/clear_stackoverflow_posts.csv'
    vocab_output_path = 'data/vocab_stackoverflow_posts.pkl'
    nrows = None

    raw_df = pd.read_csv(file_path, nrows=nrows, usecols=['Title', 'Body', 'Tags'])
    df = raw_df[~raw_df['Tags'].isna()]
    if nrows: 
        df = df.iloc[:nrows, :]
        
    df.reset_index(inplace=True)
    print(f'samples count {df.shape[0]}')
    
    df.fillna('', inplace=True)


    print('Parse tags...')
    df["Tags"] = df["Tags"].apply(lambda x: parse_tags(x))
    

    print('Tokenizing text...')
    df['Tokenize_title'] = df['Title'].apply(lambda x: preprocessor(x))
    df['Tokenize_body'] = df['Body'].apply(lambda x: preprocessor(x))
    
    print('Save dataset...')
    df.to_csv(output_path, index=True, index_label='idx', columns=['Title', 'Body', 'Tags', 'Tokenize_title', 'Tokenize_body'])
    
    print('Build vocab...')
    vocab = build_vocab_from_iterator(yield_tokens(df), specials=['<unk>', '<pad>'])
    vocab.set_default_index(vocab['<unk>'])
    
    print('Save vocab...')
    save_vocab(vocab, vocab_output_path)
    