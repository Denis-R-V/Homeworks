import pickle
import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords

from pymorphy2 import MorphAnalyzer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

import pandas as pd

tokenizer = WordPunctTokenizer()

nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))
special_char=[",",":"," ",";",".","?"]

morph = MorphAnalyzer()
snowball = SnowballStemmer(language="russian")
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

count = pickle.load(open("NLP_model/count.pkl", 'rb'))
feature_names = count.get_feature_names_out()
tfidf_transformer = pickle.load(open("NLP_model/tfidf_transformer.pkl", 'rb'))

logreg = pickle.load(open("NLP_model/model_logreg.pkl", 'rb'))
SVC_model = pickle.load(open("NLP_model/model_svc.pkl", 'rb'))
LinearSVC_model = pickle.load(open("NLP_model/model_linearSVC.pkl", 'rb'))
GradBst = pickle.load(open("NLP_model/model_RandForest.pkl", 'rb'))
RandForest = pickle.load(open("NLP_model/model_GradBst.pkl", 'rb'))
Stacking_models = pickle.load(open("NLP_model/model_Stacking_models.pkl", 'rb'))

def string_to_tfidf(phrase):
    # токенизация
    phrase_tok = tokenizer.tokenize(phrase.lower())
    # препроцессинг фразы
    phrase_lem_stem = []
    for w in phrase_tok:
        if w not in stop_words and w not in special_char:
            # лемматизация и стемминг русских слов
            word_lem_stem_rus = snowball.stem(morph.normal_forms(w)[0])
            # лемматизация и стемминг английских слов
            word_lem_stem_rus_eng = ps.stem((lemmatizer.lemmatize(word_lem_stem_rus)))
            phrase_lem_stem.append(word_lem_stem_rus_eng)
            string = (' '.join(phrase_lem_stem))
    phrase_preprocessed = [string]
    #using the count vectorizer
    phrase_words_count = count.transform(phrase_preprocessed)
    #tfidf
    phrase_words_tfidf_short = tfidf_transformer.transform(phrase_words_count)
    phrase_words_tfidf = pd.DataFrame(phrase_words_tfidf_short.T.todense(), index=feature_names, columns=[0])
    return phrase_words_tfidf.T

def predict_logreg(phrase):
    phrase_tfidf = string_to_tfidf(phrase) 
    if logreg.predict(phrase_tfidf) > 0.5:
        is_toxic = "The comment is toxic"
    else:
        is_toxic = "The comment isn't toxic"
    return is_toxic

def predict_svc(phrase):
    phrase_tfidf = string_to_tfidf(phrase) 
    if SVC_model.predict(phrase_tfidf) > 0.5:
        is_toxic = "The comment is toxic"
    else:
        is_toxic = "The comment isn't toxic"
    return is_toxic

def predict_linearSVC(phrase):
    phrase_tfidf = string_to_tfidf(phrase) 
    if LinearSVC_model.predict(phrase_tfidf) > 0.5:
        is_toxic = "The comment is toxic"
    else:
        is_toxic = "The comment isn't toxic"
    return is_toxic

def predict_GradBst(phrase):
    phrase_tfidf = string_to_tfidf(phrase) 
    if GradBst.predict(phrase_tfidf) > 0.5:
        is_toxic = "The comment is toxic"
    else:
        is_toxic = "The comment isn't toxic"
    return is_toxic

def predict_RandForest(phrase):
    phrase_tfidf = string_to_tfidf(phrase) 
    if RandForest.predict(phrase_tfidf) > 0.5:
        is_toxic = "The comment is toxic"
    else:
        is_toxic = "The comment isn't toxic"
    return is_toxic

def predict_Stacking_models(phrase):
    phrase_tfidf = string_to_tfidf(phrase) 
    if Stacking_models.predict(phrase_tfidf) > 0.5:
        is_toxic = "The comment is toxic"
    else:
        is_toxic = "The comment isn't toxic"
    return is_toxic

phrase = str(input())
print(f"Фраза: {phrase}")
print("\nКлассификация с помощью логистической регрессии:")
print(predict_logreg(phrase))
print("\nКлассификация с помощью SVC:")
print(predict_svc(phrase))
print("\nКлассификация с помощью LinearSVC:")
print(predict_linearSVC(phrase))
print("\nКлассификация с помощью градиентного бустинга:")
print(predict_RandForest(phrase))
print("\nКлассификация с помощью случайного леса:")
print(predict_GradBst(phrase))
print("\nКлассификация с помощью стэкинга:")
print(predict_Stacking_models(phrase))