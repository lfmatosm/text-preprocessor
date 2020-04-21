from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import spacy
import re


#Makes basic, generic preprocessing on documents corpus
class Preprocessor:
    def __init__(self, language="en", lemmatize_activated=True):
        self.__lemmatize_activated = lemmatize_activated

        self.__nlp = spacy.load("pt_core_news_sm") if (language == "pt") else spacy.load("en")
        self.__stop_words = stopwords.words("portuguese") if (language == "pt") else stopwords.words("english")


    #Removes newline chars from each document
    def remove_newlines_and_single_quotes(self, texts):
        remove_newlines = lambda text: re.sub(r'\s+', ' ', text)

        remove_single_quotes = lambda text: re.sub("\'", "", text)

        without_newlines = map(remove_newlines, texts)

        return map(remove_single_quotes, without_newlines)


    #Removes documents with size of less than n words
    def filter_documents_with_less_than(self, tokenized_documents, min_words=5):
        return list(filter(lambda tokenized_text: len(tokenized_text) > min_words, tokenized_documents))


    #Removes stopwords
    def remove_stopwords(self, texts):
        return [[word for word in doc if word not in self.__stop_words] for doc in texts]


    #Transforms each word into its base form. e.g. 'fazendo' becomes 'fazer'
    def lemmatize(self, documents, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        texts_out = []
        for document in documents:
            doc = self.__nlp(" ".join(document)) 
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out


    #Removes undesired chars (like newlines), break documents into words lists,
    # remove stopwords and smaller length documents from original data
    def preprocess(self, data):
        data_without_newlines = self.remove_newlines_and_single_quotes(data)

        print("Newlines and single-quotes removed from documents")

        #Breaks each document into a list of words
        tokenize = lambda texts: [(yield simple_preprocess(text, deacc=True)) for text in texts]

        tokenized_data = tokenize(data_without_newlines)

        print("Tokenized documents.")

        data_without_stopwords = self.remove_stopwords(tokenized_data)

        print("Stopwords removed.")

        if (self.__lemmatize_activated):

            lemmatized_data = self.lemmatize(data_without_stopwords)

            print("Lemmatized data.")

            return lemmatized_data

        return data_without_stopwords
