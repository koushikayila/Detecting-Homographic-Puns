import xml.etree.ElementTree as ET
import numpy as np
import gensim
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import spacy, fasttext

# model = gensim.models.Word2Vec.load_word2vec_format('/home/koushik/Desktop/NLP/FunWithPun-master/sample/GoogleNews-vectors-negative300.bin', binary=True)
text = open('./test.txt','r');
gold = open('./a.gold','r');
gold = gold.read();
goldline = gold.split('\n');
text_content = text.read();
listposold = '';
lines_list = text_content.split('\n');
final = {};
ans = '';
our_answer = {};

for i in lines_list:
     listpos = i.split('_')[1];
     if(listpos != listposold):
         ans = ''
     temp1 = i.split('>');
     ans = ans + temp1[1].split('<')[0] + ' ';
     final[int(listpos)] = ans;
     listposold = listpos;

goldnum = [];
goldline = goldline[:-1]
for i in goldline:
    number = i.split('_')[1].split('\t');
    num = number[1];
    goldnum.append(num);

words = [];

for i in sorted(final):
    words.append(final[i]);

our_answer = dict(zip(words,goldnum));
# fi = open('train.txt', 'w')
# for s in our_answer:
#     fi.write(s.strip()+'\n')
# model = fasttext.skipgram('train.txt', 'model', dim=25)
model = fasttext.load_model('model.bin')
#print(model['hello'])
X =[]

for s in our_answer:
    avg = [0]*25
    s = s.split(' ')
    for j in range(len(s)):
        for k in range(len(avg)):
            avg[k] +=model[s[j]][k]
    for j in range(len(avg)):
        avg[j] /= len(s)
        X.append(avg)
Y = list(our_answer.values())
# print(X[4]);
# print(Y[4]);
