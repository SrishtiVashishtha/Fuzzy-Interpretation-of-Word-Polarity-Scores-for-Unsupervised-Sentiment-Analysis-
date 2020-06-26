import nltk
from nltk.corpus import sentiwordnet as swn

from sklearn.datasets import load_files
reviews = load_files("C:/MyData/PythonPractice/Pang Lee Movie Review Dataset/", shuffle = False) #default: shuffle = True
text,labels = reviews.data, reviews.target

print("Number of documents in data: {}".format(len(text)))
print("Number of Labels in data: {}".format(len(labels)))  #0: negative, 1: positive

senti=[]
num_list=[]

for k in range(len(text)):

    print(k)
    review=text[k].decode()
#    print(review)
    allwords = nltk.word_tokenize(review)
    allwords=[word.lower() for word in allwords if word.isalpha()]
#    print(allwords)
    wnl = nltk.WordNetLemmatizer()
    allwords=[wnl.lemmatize(word) for word in allwords]
#    print(allwords)
    tagged = nltk.pos_tag(allwords)
#    print(tagged)
    taglist=[]
    
    score_list_pos=[]
    score_list_neg=[]
    sentence_sentiment=[]

    for i in range(len(allwords)):
        t=tagged[i][1]
        if t.startswith('NN'):
            newtag='n'
        elif t.startswith('JJ'):
            newtag='a'
        elif t.startswith('V'):
            newtag='v'
        elif t.startswith('R'):
            newtag='r'
        else:
            newtag='' 
        taglist.append(newtag)
        if(newtag!=''):    
            synsets = list(swn.senti_synsets(allwords[i], newtag))
			#Getting average       
#            scorepos=scoreneg=0
            if(len(synsets)>0):
                pos=[]
                neg=[]
                for syn in synsets:
                    scorepos=syn.pos_score()
                    scoreneg=syn.neg_score()
#                    print(scorepos)
#                    print(scoreneg)
                    pos.append(scorepos)
                    neg.append(scoreneg)

                nscorepos=sum(pos)/len(synsets)
                nscoreneg=sum(neg)/len(synsets)
#                print(nscorepos)
#                print(nscoreneg)

        score_list_pos.append(nscorepos)
        score_list_neg.append(nscoreneg)
#        print(score_list_pos)
#        print(score_list_neg)
            
    card_pos=sum(score_list_pos) 
    card_neg=sum(score_list_neg)
    
#    print(card_pos)
#    print(card_neg)    

    if card_pos>=card_neg:
        num_list.append(1)  # positive sentiment
    else:
        num_list.append(0)  # negative sentiment

count=0
for j in range(len(text)):
    if num_list[j]==labels[j]:
        count=count+1
acc=(count/len(text))*100
print("Accuracy: "+str(acc)+"%")
