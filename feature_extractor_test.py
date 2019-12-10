import os
import json
import spacy
from collections import OrderedDict
from model_factory import CFG_FEATURES, CFG_LANG
import pandas as pd

from feature_extractor.extractor_facade import FeatureExtractor
from feature_extractor.extractor_length import ExtractorLength
from feature_extractor.extractor_sentiment import ExtractorSentiment
from feature_extractor.extractor_linguistic import ExtractorLinguistic
from feature_extractor.extractor_bow import ExtractorBow

DIR_ROOT = os.path.dirname(__file__)
CFG_ENGLISH = os.path.join(DIR_ROOT, 'configs/english.json')
DIR_MODELS = os.path.join(DIR_ROOT, 'models/')
DIR_MODELS_EN = os.path.join(DIR_MODELS, 'english/')
cfg = json.load(open(CFG_ENGLISH), object_pairs_hook=OrderedDict)

feature_vector = OrderedDict()

#sentiment
result_sentiment_1, result_sentiment_2 = ExtractorSentiment.extract('en', {"text":"I hate you, I will kill you"}, cfg["irrelevant"][CFG_FEATURES]["sentiment"])
print("##########sentiment##########################")
print(result_sentiment_1)
print(result_sentiment_2)
feature_vector.update(result_sentiment_2)

#linguistic
result_linguistic =  ExtractorLinguistic.extract('en', {"text":"@messenger @facebook guys, could you finally let me back into my account? i already came back to the usa and you st "}, cfg["irrelevant"][CFG_FEATURES]["linguistic"])
print("##########linguistic##########################")
print(result_linguistic)
feature_vector.update(result_linguistic)

#bow
# result_bow = ExtractorBow.extract('en', {"text":"This is a good app!","processed_tweet":"This is a good app!"}, cfg["irrelevant"][CFG_FEATURES]["bow"])
# print("bow##########################")
# print(result_bow)
# feature_vector.update(result_bow)


#preprocess
print("##########preprocess##########################")
result_preprocess = FeatureExtractor(cfg["irrelevant"], "en", {"text":"@burasto @tha_rami @gmail lena gets ads like this shit on ios all the time and i never ever saw it on the android app."})
print(result_preprocess.preprocess())


#dataframe
data_vector = pd.DataFrame([feature_vector], columns=feature_vector.keys())
print("##########data vectora####################")
print(data_vector)

#length
result_length =  ExtractorLength.extract('en', {"processed_tweet":"This is a good app!"}, cfg["irrelevant"][CFG_FEATURES]["length"])
print(result_length["processed_tweet_length"])
feature_vector.update(result_length)

#combine
# dataset = [
#         {"text": "@Op1 buongiorno ho scritto ieri un dm nessuna risposta!"},
#         {"text": "@Op1 voglio dire grazie pubblicamente al vostro operatore xxxxxxx (lavorava ieri pomer.) che in due minuti ha risolto tutto. Tks"},
#         {"text": "@Op1 buongiorno sto partendo per le Grecia, posso utilizzare il mio piano tariffario senza spendere nulla? nGrazie"},
#         {"text": "@Op1 √à prevista qualche offerta con almeno 10GB con uno costo di attivazione inferiore?"},
#         {"text": "Cara @Op1 impossibile da tempo parlare con vs operatore e sto pagando giochi nn richiesti ¬†nPasso alla concorrenza o mi contattate subito?"},
#         {"text": "@Op1 sono senza linea (cellulare) da ieri.Posso usare solo il wifi di casa ma non posso n√© chiamare n√© ricevere. Numero mandato in DM"}]
# for tweet in dataset:
# 	tweet_vector = FeatureExtractor(cfg["irrelevant"], "en", tweet)
# 	tweet_vectors = tweet_vectors.append(tweet_vector.data_vector)

data_reader = pd.read_csv('../data/trainingset_450.csv', encoding = "utf-8")
print(data_reader)
print(data_reader['text'][0])
tweet_vectors = pd.DataFrame()
for index, row in data_reader.iterrows():
	tweet = {"text":data_reader['text'][index]}
	tweet_vector = FeatureExtractor(cfg["irrelevant"], "en", tweet)
	tweet_vectors = tweet_vectors.append(tweet_vector.data_vector)

tweet_vectors = tweet_vectors.reset_index(drop=True)
print(tweet_vectors)
tweet_vectors.to_csv('result_vectors.csv')







