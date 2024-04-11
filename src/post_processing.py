import os
import json
import datetime
import nltk
from nltk.corpus import stopwords
import string
from utils.utils import preprocess_text


def idk(test_labels, retrieved_items, D, test_data, th = 1.5917805):
    print ("D.shape", D.shape)
    D = D.tolist()
    bad_retrieved_items = [(v,l,n) for (v,l,m,n) in zip(test_labels, retrieved_items, D, test_data) if min(m)>=th]
    bad_retrieved_labels = [i[0] for i in bad_retrieved_items]
    bad_retrieved_query = [i[2] for i in bad_retrieved_items]
    bad_retrieved_items = [i[1] for i in bad_retrieved_items]

    good_retrieved_items = [(v,l, n) for (v,l,m, n) in zip(test_labels, retrieved_items, D, test_data) if min(m)<th]
    good_retrieved_labels = [i[0] for i in good_retrieved_items]
    good_retrieved_query = [i[2] for i in good_retrieved_items]
    good_retrieved_items = [i[1] for i in good_retrieved_items]
    print ("good_retrieved_items", len(good_retrieved_items))



    k = len(good_retrieved_items[0])
    
    bad_retrieved_items = [["None"]*k for i in range(len(bad_retrieved_items))]
    retrieved_items_ = good_retrieved_items + bad_retrieved_items
    test_labels_ = good_retrieved_labels + bad_retrieved_labels
    test_query = good_retrieved_query + bad_retrieved_query

    retrieved_items_clean = []
    for i in retrieved_items_:
        temp = []
        for x in i:
            temp.append(preprocess_text(x))
        retrieved_items_clean.append(temp)

    test_labels_clean = [x if isinstance(x, str) else "None" for x in test_labels_]
    test_labels_clean = [preprocess_text(x) for x in test_labels_clean]

    return test_labels_clean, retrieved_items_clean, test_query