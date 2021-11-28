import os
import random
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sumeval.metrics.rouge import RougeCalculator

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from summa.summarizer import summarize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer

# We can use ratio like a hyperparameter to control the length of the summary
def textrank(file, ratio=0.02):
    with open(file, 'r', errors='ignore') as f:
        doc = f.read()
    return summarize(doc, ratio=ratio)

def sumy_textrank(file, ratio=0.02):
    with open(file, 'r', errors='ignore') as f:
        doc = f.read().split('\n')
    n = len(doc)
    parser = PlaintextParser(doc, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summaryinit = summarizer(parser.document, round(ratio * n))
    summary = ''
    for i in summaryinit:
        summary += str(i)
    summary = summary.replace(',', '')
    return summary

def centroid(file, binary=False, ratio=0.02):
    with open(file, 'r', errors='ignore') as f:
        doc = f.read().split('\n')
    n = len(doc)
    if binary:
        tfidfv = TfidfVectorizer(stop_words=stopwords.words('english'), binary=True, use_idf=False, norm=False)
    else:
        tfidfv = TfidfVectorizer(stop_words=stopwords.words('english'))
    tfidfmat = tfidfv.fit_transform(doc)
    features = tfidfv.get_feature_names()
    # centroiddf = pd.DataFrame(tfidfmat.todense(), index=doc, columns=features)
    top_sents = np.concatenate((np.asarray(doc).reshape(-1, 1), tfidfmat.todense().sum(axis=1)), axis=1)
    top_df = pd.DataFrame(top_sents, columns=['Sentence', 'TF-IDF-Score'])
    top_df['TF-IDF-Score'] = pd.to_numeric(top_df['TF-IDF-Score'], errors='coerce')
    top_df.sort_values(by='TF-IDF-Score', inplace=True, ascending=False)
    top_df['Sentence'] = top_df['Sentence'].replace(',', '')
    return '\n'.join(top_df['Sentence'][:round(ratio * n)])

def lsa(file, ratio=0.02):
    with open(file, 'r', errors='ignore') as f:
        doc = f.read().split('\n')
    n = len(doc)
    parser = PlaintextParser(doc, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summaryinit = summarizer(parser.document, round(ratio * n))
    summary = ''
    for i in summaryinit:
        summary += str(i)
    summary = summary.replace(',', '')
    return summary

def lexrank(file, ratio=0.02):
    with open(file, 'r', errors='ignore') as f:
        doc = f.read().split('\n')
    n = len(doc)
    parser = PlaintextParser(doc, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summaryinit = summarizer(parser.document, round(ratio * n))
    summary = ''
    for i in summaryinit:
        summary += str(i)
    summary = summary.replace(',', '')
    return summary

def luhn(file, ratio=0.02):
    with open(file, 'r', errors='ignore') as f:
        doc = f.read().split('\n')
    n = len(doc)
    parser = PlaintextParser(doc, Tokenizer("english"))
    summarizer = LuhnSummarizer()
    summaryinit = summarizer(parser.document, round(ratio * n))
    summary = ''
    for i in summaryinit:
        summary += str(i)
    summary = summary.replace(',', '')
    return summary

def init_matrix(file):
    with open(file, 'r', errors='ignore') as f:
        doc = f.read()
    vocabulary = list(set(word_tokenize(doc)))
    vocab_matrix = pd.DataFrame(0, index=vocabulary, columns=vocabulary)
    return vocab_matrix

def store_positions_info(file, maxlength):
    with open(file, 'r', errors='ignore') as f:
        doc = f.read()
    vocabulary = list(set(word_tokenize(doc)))
    infodict = pd.DataFrame(0, index=vocabulary, columns=range(1, maxlength + 1))
    for sentence in sent_tokenize(doc):
        wordlist = word_tokenize(sentence)
        for idx, word in enumerate(wordlist, 1):
            infodict[idx][word] += 1
    return infodict

def make_edges(file, vocab_matrix):
    with open(file, 'r', errors='ignore') as f:
        doc = f.read()
    for sentence in sent_tokenize(doc):
        wordlist = word_tokenize(sentence)
        for i in range(len(wordlist) - 1):
            vocab_matrix[wordlist[i]][wordlist[i + 1]] += 1
    return vocab_matrix

def importance_scores(file, vocab_matrix, ends=False, mid=False):
    sentence_scores = []
    sentence_scores_mid = []
    sentence_scores_ends = []
    with open(file, 'r', errors='ignore') as f:
        doc = f.read()
    for sentence in sent_tokenize(doc):
        score = 0
        scoremid = 0
        scoreends = 0
        wordlist = word_tokenize(sentence)
        sentlength = len(wordlist)
        weights_middle = list(np.linspace(0, 1, sentlength // 2))
        weights_middle.extend(list(np.linspace(1, 0, sentlength // 2)))
        if sentlength % 2 == 0:
            del weights_middle[sentlength // 2]
        weights_ends = list(np.linspace(1, 0, sentlength // 2))
        weights_ends.extend(list(np.linspace(0, 1, sentlength // 2)))
        if sentlength % 2 == 0:
            del weights_ends[sentlength // 2]
        for idx in range(len(wordlist) - 1):
            score += vocab_matrix[wordlist[idx]][wordlist[idx + 1]]
            scoremid += vocab_matrix[wordlist[idx]][wordlist[idx + 1]] * weights_middle[idx]
            scoreends += vocab_matrix[wordlist[idx]][wordlist[idx + 1]] * weights_ends[idx]
        score /= len(wordlist)
        scoremid /= len(wordlist)
        scoreends /= len(wordlist)
        sentence_scores.append([sentence, score])
        sentence_scores_mid.append([sentence, scoremid])
        sentence_scores_ends.append([sentence, scoreends])
    imp_scores = pd.DataFrame(sentence_scores, columns=['Sentence', 'Importance-Score'])
    imp_scores_mid = pd.DataFrame(sentence_scores_mid, columns=['Sentence', 'Importance-Score'])
    imp_scores_ends = pd.DataFrame(sentence_scores_ends, columns=['Sentence', 'Importance-Score'])
    if ends:
        return imp_scores_ends
    elif mid:
        return imp_scores_mid
    else:
        return imp_scores

# The BigramRank algorithm
def phraserank(file, ratio=0.02, ends=False, mid=False):
    vocab_matrix = init_matrix(file)
    vocab_matrix = make_edges(file, vocab_matrix)
    # print(vocab_matrix)
    
    maxlength = 0
    with open(file, 'r', errors='ignore') as f:
        doc = f.read()
    n = len(sent_tokenize(doc))
    for sentence in sent_tokenize(doc):
        sentlength = len(word_tokenize(sentence))
        if maxlength < sentlength:
            maxlength = sentlength
    # print(maxlength)
    infodict = store_positions_info(file, maxlength)
    top_df = importance_scores(file, vocab_matrix, ends, mid)
    top_df['Importance-Score'] = pd.to_numeric(top_df['Importance-Score'], errors='coerce')
    top_df.sort_values(by='Importance-Score', inplace=True, ascending=False)
    top_df['Sentence'] = top_df['Sentence'].replace(',', ' ')
    # print(top_df)
    return '\n'.join(top_df['Sentence'][:round(ratio * n)])

def next_neighbour(infodict, pos, rank) -> (str):
    infodict = infodict.rename_axis('Words')
    infodict = infodict[infodict[pos] > 0]
    temp = infodict[pos].sort_values(ascending=False).reset_index().rename(columns={'index':'Words'})
    return temp.iloc[rank - 1]

def compute_maxscore(infodict: pd.DataFrame, maxlength: int) -> int:
    maxsc = 0
    wordset = set()
    for pos in range(1, maxlength + 1):
        df = infodict[infodict[pos] > 0]
        temp = df[pos].sort_values(ascending=False).reset_index().rename(columns={'index':'Words'})
        n = len(temp.index)
        rank = 1
        # print(temp['Words'].iloc[rank])
        while temp['Words'].iloc[rank - 1] in wordset and rank < n:
            rank += 1
        wordset.add(temp['Words'].iloc[rank - 1])    
        maxsc += temp[pos].iloc[rank - 1]
    return maxsc

def traverserank(file, ratio=0.02, score_ratio=0.5):
    vocab_matrix = init_matrix(file)
    vocab_matrix = make_edges(file, vocab_matrix)
    # print(vocab_matrix)
    
    maxlength = 0
    with open(file, 'r', errors='ignore') as f:
        doc = f.read()
    n = len(sent_tokenize(doc))
    for sentence in sent_tokenize(doc):
        sentlength = len(word_tokenize(sentence))
        if maxlength < sentlength:
            maxlength = sentlength
    # print(maxlength)
    infodict = store_positions_info(file, maxlength)
    maxscore = compute_maxscore(infodict, maxlength)
    target_score = score_ratio * maxscore
    summarystr = str()
    sentence_score = 0
    pos = 1
    # print(target_score)
    while sentence_score < target_score and pos <= maxlength:
        # print(sentence_score)
        rank = 1
        n = len(infodict[infodict[pos] > 0])
        temp = next_neighbour(infodict, pos=pos, rank=rank)
        # print(temp['Words'])
        # print(summarystr)
        while temp['Words'] in summarystr and rank < n:
            rank += 1
            temp = next_neighbour(infodict, pos=pos, rank=rank)
        summarystr += temp['Words'] + ' '
        sentence_score += temp[pos]
        pos += 1
    return summarystr

rouge = RougeCalculator(stopwords=True, lang='en')
dirs = os.listdir('../data/Opinosis/summaries-gold')
traindir = os.listdir('../data/Opinosis/topics')

def printsamples(n):
    with open(os.path.join('../data/Opinosis/topics', traindir[n]), 'r', errors='ignore') as f:
        doc = f.read()
    summarytxt = textrank(os.path.join('../data/Opinosis/topics', traindir[n]))
    nametxt = "TextRank"
    summarystxt = sumy_textrank(os.path.join('../data/Opinosis/topics', traindir[n]))
    namestxt = "TextRank(Sumy)"
    summaryc = centroid(os.path.join('../data/Opinosis/topics', traindir[n]))
    namec = "Centroid"
    summarylsa = lsa(os.path.join('../data/Opinosis/topics', traindir[n]))
    namelsa = "LSA"
    summarylex = lexrank(os.path.join('../data/Opinosis/topics', traindir[n]))
    namelex = "LexRank"
    summarylu = luhn(os.path.join('../data/Opinosis/topics', traindir[n]))
    namelu = "Luhn"
    summarypr = phraserank(os.path.join('../data/Opinosis/topics', traindir[n]))
    namepr = "BigramRank"
    summarypre = phraserank(os.path.join('../data/Opinosis/topics', traindir[n]), ends=True)
    namepre = "BigramRank (Ends)"
    summaryprm = phraserank(os.path.join('../data/Opinosis/topics', traindir[n]), mid=True)
    nameprm = "BigramRank (Middle)"
    summarydfs = traverserank(os.path.join('../data/Opinosis/topics', traindir[n]))
    namedfs = "TraverseRank"
    algols = [[summarytxt, nametxt], [summaryc, namec], [summarylsa, namelsa], [summarylex, namelex], [summarylu, namelu], [summarypr, namepr], [summarypre, namepre], [summaryprm, nameprm], [summarydfs, namedfs]]
    algomyls = [[summarypr, namepr], [summarypre, namepre], [summaryprm, nameprm], [summarydfs, namedfs]]
    print("Article Number: {}\n".format(n))
    print('The article is\n-----------------------------------------------------\n{}'.format(doc))
    for algo in algols:
        print('\nThe summary generated by {} Algorithm is\n-----------------------------------------------------\n{}'.format(algo[1], algo[0]))

printsamples(random.randint(1, 50))
# for i in range(len(traindir)):
#     print('\nArticle {}'.format(i))
#     printsamples(i)