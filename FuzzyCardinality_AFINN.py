from afinn import Afinn
import nltk
af = Afinn(emoticons=True)

from sklearn.datasets import load_files
reviews = load_files("C:/MyData/PythonPractice/Pang Lee Movie Review Dataset/", shuffle = False) #default: shuffle = True
text,labels = reviews.data, reviews.target

print("Number of documents in data: {}".format(len(text)))
print("Number of Labels in data: {}".format(len(labels)))  #0: negative, 1: positive

senti=[]
senti2=[]

num_list=[]
for k in range(len(text)):
    
    print(k)
    review=text[k].decode()
#    print(review)
    afscore=af.score(review)
    
        # Tokenization
    words = nltk.word_tokenize(review)
#    print(words)
    
     # compute sentiment scores (polarity) and labels
    sentiment_scores = [af.score(word) for word in words]
    sentiment_category = ['positive' if score >= 0 
                              else 'negative'  
                                      for score in sentiment_scores]
    
    positive=[]
    negative=[]
    
    for score in sentiment_scores:
        if score>=0:
            positive.append(score/5)  # to normalize the AFINN score divide it by 5, because it lies between -5 to 5
        elif score<0:
            negative.append(score/5)  # to normalize the AFINN score divide it by 5, because it lies between -5 to 5
            
#    print(positive)
#    print(negative)
            
##    print("\n Positive Fuzzy Cardinality for each  review :")
    card_posscore=sum(positive)   #add up all pos scores- fuzzy cardinality
#    print(posscore)
#
##    print("\n Negative Fuzzy Cardinality for each  review :")
    card_negscore=sum(negative)  #add up all neg scores- fuzzy cardinality
#    print(negscore)
                
    if card_negscore<0:
        card_negscore=-card_negscore    # to make it absolute value
#    print(negscore)

    if card_posscore>=card_negscore:
        num_list.append(1)  # positive sentiment
    else:
        num_list.append(0)  # negative sentiment
    
 
count=0
for j in range(len(text)):
    if num_list[j]==labels[j]:
        count=count+1
acc=(count/len(text))*100
print("Accuracy: "+str(acc)+"%")  # AFINN CARDINALITY
