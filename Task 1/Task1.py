import xml.etree.ElementTree as ET
import numpy as np
import gensim
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import spacy, fasttext
import sys
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
X = np.array(X)

class_mean = {}
class_var = {}
class_ind ={}
labels = []
cnt = 0
for i in range(len(X)):
    clas = str(Y[i])
    if clas not in class_ind.keys():
        class_mean[clas]=[]
        class_var[clas]=[]
        class_ind[clas]=[]
        labels.append(clas)
    class_ind[clas].append(cnt)
    cnt+=1

images = np.array(X)

images = np.array(images)

image_mean =images - np.mean(images,axis=0)
# print(len(images))



cov_image = np.dot(image_mean.T,image_mean)
image_lam, image_vec = np.linalg.eig(cov_image)

# print(image_vec)

indi = np.argsort(-image_lam)


image_vec = image_vec[:,indi]

# sorted_vec = image_vec[:,np.flip(indi,axis=0)]


N = 10

n_vec = image_vec[:,0:N]
n_data = np.matmul(n_vec.T,images.T)
n_data = n_data.T

for i in labels:
    dattt = np.array(n_data[class_ind[i],:])
    class_mean[i].append(np.mean(dattt,axis=0))
    class_var[i].append(np.var(dattt,axis=0))

fp1 = open(sys.argv[1])

test = fp1.readlines()

flag = 0

for s in test:
    avg = [0]*25
    s = s.split(' ')
    for j in range(len(s)):
        for k in range(len(avg)):
            avg[k] +=model[s[j]][k]
    for j in range(len(avg)):
        avg[j] /= len(s)

    test_image = avg
    array = np.array(test_image)
    array = array.flatten()


    test_data = np.matmul(n_vec.T,array.T)
    test_data = test_data.T
    max_p = 0
    ans = ''
    for clas in labels:
        prob  = 1
        mean = class_mean[clas][0]
        var  = class_var[clas][0]


        for j in range(len(test_data)):
            val = 1/np.sqrt(2*np.pi*var[j])
            prob = prob*(np.exp(-abs(np.square(test_data[j]-mean[j])/var[j])))*val
        # prob = prob * class_count[clas]
        if prob > max_p:
            max_p = prob
            ans = clas

    if ans == '1':
        flag =1
        break

if flag == 1:
    print('YES; the given context contains a pun')
else:
print('NO; the given context does not contain a pun')
