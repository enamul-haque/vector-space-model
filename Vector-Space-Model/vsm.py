# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 01:41:30 2017

@author: Md. Enamul Haque
"""
import pandas as pd
import numpy as np
import math
from nltk import stem
import nltk
import re
from string import punctuation
from numpy import linalg as LA
import os
import time
import sys



def prepare_query():
    global qindex, qdesc    
    # modify this path where you put cran.all file
    query_file = direct+'/cran/cran.qry'
    infile = open(query_file, 'r')

    text = infile.readlines()
    #print(text)
    

    qindex = [] # title of the document
    qdesc = [] # author or institute of the document
    qd = []
    active=qindex
    """ activate the handle and save data for each handle until a new title is found
    """
    for j in text:
        
        if j.startswith('.I'):
            if len(qd) > 0:
                qdesc.append(qd)
                qd = []
            qindex.append(int(j[3:].strip('\n')))
            #print(qindex)
        elif j.startswith('.W'):
            active = qd
        else:
            if j[:2] !='.I':
                active.append(j.strip('\n'))
    qdesc.append(qd)
    
def remove_punctuation(s):
    """
    remove punctuations: !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~
    """
    s = re.sub('[%s]' % re.escape(punctuation), '', str(s))
    s = re.sub( '\s+', ' ', s ).strip()
    
    return s
    
def stemmer(d):
    # stem tokens of each documents
    stemmer = stem.PorterStemmer()
    for k in range(len(d)):
        temp = stemmer.stem(d[k])
        d[k] = temp
    return d
    
def remove_stopwords(text):
    """
    stop words removal
    """
    new_text = []
    tokens = nltk.word_tokenize(text.lower())
    
    for i in range(len(tokens)):
        if (tokens[i] not in stopwords):
            new_text.append(tokens[i])
         
    return new_text
    
def tokenize(document):
    """Returns a list whose elements are the separate terms in
    document.  Something of a hack, but for the simple documents we're
    using, it's okay.  Note that we case-fold when we tokenize, i.e.,
    we lowercase everything."""
    characters = " .,!#$%^&*();:\n\t\\\"?!{}[]<>"
    terms = document.lower().split()
    return [term.strip(characters) for term in terms]
    
def default_query_processing():
    
    prepare_query()
    d = data_matrix
    d.set_index('doc_id', inplace=True)
    
    f = open('scores_output.txt','w')
    f.write( 'Rank'+'\t'+'Query#'+'\t'+'Scores'+'\t'+'Document#'+'\t'+'Snippet of text#'+'\t'+'Valid ranks count'+'\n' )
        
    for j in range(len(qdesc)):
        # remove punctuation from each documents
        query = remove_punctuation(qdesc[j])
            
        # remove stopwords from each documents
        query = remove_stopwords(query)
            
        # stem
        query = stemmer(query)
            
        q_mat = pd.DataFrame(columns={'index_terms','q'})
        q_mat.index_terms = unique_words
        q_mat.fillna(0,inplace=True)
        q_mat.set_index('index_terms', inplace=True)
        
        # intersect query and index terms
        valid_q = list(set(query).intersection(list(A.index)))
        
        for m in range(len(valid_q)):
            for n in range(len(A)):
                if valid_q[m] == A.index[n]:
                    q_mat.q[n] = 1
            
        q_mat.q = q_mat.q * A.IDF
        scores = np.zeros(N+2)
        scores[0] = -5555555
        """
        cosine similarity function:
        if any of the denominator or numerator is zero the score is set to 0
        """
        for i in range(1,N+2):
            vec_dot = np.dot(q_mat.q, A['doc_'+str(i)])
            vec_norm = (LA.norm(q_mat.q) * LA.norm(A['doc_'+str(i)]))
            if vec_dot == 0 or vec_norm == 0:
                scores[i] = 0.0
            if vec_dot > 0 or vec_norm > 0:
                scores[i] = vec_dot / vec_norm 
                
        
        top_scores = sorted(scores,reverse=True)
        top_n_scores = top_scores[:K]
        top_docs = np.array(np.argsort(scores))
        top_docs = top_docs[::-1] # reverse
        top_n_docs = top_docs[:K] 
        # total numbe of valid score
        valid_scores = np.count_nonzero(scores)
        
        
        # write data
        for p in range(K):
            f.write( str(p+1)+'\t'
            +str(qindex[j])+'\t'
            +str(round(top_n_scores[p],2))+'\t'
            +str(top_n_docs[p])+'\t'
            +str(d.loc[top_n_docs[p]].title)+'\t'
            + str(valid_scores)+'\n' )
        f.write('*********************************************************************************\n')
    
    f.close()

def user_query_processing(user_query):
    
        d = data_matrix
        # remove punctuation from each documents
        query = remove_punctuation(user_query)
            
        # remove stopwords from each documents
        query = remove_stopwords(query)
            
        # stem
        query = stemmer(query)
            
        q_mat = pd.DataFrame(columns={'index_terms','q'})
        q_mat.index_terms = unique_words
        q_mat.fillna(0,inplace=True)
        q_mat.set_index('index_terms', inplace=True)
        
        # intersect query and index terms
        valid_q = list(set(query).intersection(list(A.index)))
        
        for m in range(len(valid_q)):
            for n in range(len(A)):
                if valid_q[m] == A.index[n]:
                    q_mat.q[n] = 1
            
        q_mat.q = q_mat.q * A.IDF
        #print(q_mat.q)
        scores = np.zeros(N+2)
        # the first position is not needed, so keeping an invalid value there
        scores[0] = -5555555
        
        """
        cosine similarity function:
        if any of the denominator or numerator is zero the score is set to 0
        """
        for i in range(1,N+2):
            vec_dot = np.dot(q_mat.q, A['doc_'+str(i)])
            vec_norm = (LA.norm(q_mat.q) * LA.norm(A['doc_'+str(i)]))
            if vec_dot == 0 or vec_norm == 0:
                scores[i] = 0.0
            if vec_dot > 0 or vec_norm > 0:
                scores[i] = vec_dot / vec_norm 
                
        
        top_scores = sorted(scores,reverse=True)
        top_n_scores = top_scores[:K]
        top_docs = np.array(np.argsort(scores))
        top_docs = top_docs[::-1] # reverse
        top_n_docs = top_docs[:K]
        
        print('Rank'+'\t'+'Scores'+'\t'+'Document Number'+ '\t'+ 'Document snippet'+'\n' )
        for p in range(K):
            print( str(p+1)+'\t'
            +str(round(top_n_scores[p],3))+'\t'
            +str(top_n_docs[p])+'\t'
            +str(d.loc[top_n_docs[p]].title)+'\n' )
       
if __name__ == "__main__":
    
    auto=1 # 1= default query from file, auto=0 means user provides query
    # K is the number of results you want to get for each ranking
    global K
    K = 10
    
    start = time.clock()
    # stop words directory
    global direct
    direct = os.getcwd()
    stopwords_dir = direct+'/stopwords/'
    document_dir = direct+'/documents/'
    save_to = direct+'/cran_docs/'
    
    # load stopwords into stopwords
    all_stops = open(stopwords_dir+'english.stop.txt','r').readlines()
    stopwords = []
    for w in all_stops:
        stopwords.append(w.rstrip())
        
    documents = os.listdir(document_dir)
    all_doc_id = []
    all_title = []
    all_author = []
    all_bib = []
    all_desc = []

    for doc in documents:
        content = open(document_dir+doc,'r')
        text = content.readlines()
        #print(text)

        title = []
        doc_id = []
        author = []
        bib = []
        desc = []
        active = title
        """ activate the handle and save data for each handle until a new title is found
        """
        for j in text:
            
            if j.startswith('.I'):
                doc_id = j[2:]
            elif j.startswith('.T'):
                active = title
            elif j.startswith('.A'):
                active = author
            elif j.startswith('.B'):
                active = bib
            elif j.startswith('.W'):
                active = desc
            else:
                if j[:2] != '.I':
                    active.append(j.rstrip())
        
        all_doc_id.append(int(doc_id.rstrip()))
        all_title.append(title)
        all_author.append(author)
        all_bib.append(bib)
        all_desc.append(desc)
        
    # create a data matrix with five columns
    global data_matrix
    data_matrix = pd.DataFrame(columns=['doc_id', 'title', 'author', 'bib', 'desc'])
    # insert columnwise data into the matrix
    data_matrix['doc_id'] = all_doc_id
    data_matrix['title'] = all_title
    data_matrix['author'] = all_author
    data_matrix['bib'] = all_bib
    data_matrix['desc'] = all_desc
    

    
    """
    1. remove punctuations, stop words and unnecessary whitespaces
    2. stem using porter stemmer
    """
    data_matrix.set_index(['doc_id']) # set doc_id field as index column
    
    
    for i in range(len(data_matrix)):
        
        # remove punctuation of i-th title of the data matrix and assign it to t
        t = remove_punctuation(data_matrix['title'][i])
        
        # go to the i-th title and assign new t 
        data_matrix.ix[i,'title'] = t
        
        # remove punctuation from each documents
        d = remove_punctuation(data_matrix['desc'][i])
        
        # remove stopwords from each documents
        d = remove_stopwords(d)
        
        # stem
        d = stemmer(d)
        
        # update each documents using tokens/words
        data_matrix.set_value(i,'desc',d)
        #data_matrix.ix[i,'desc'] = d
        f = open(save_to+str(data_matrix['doc_id'][i])+'.txt', 'w')
        f.write(str(data_matrix['desc'][i])[1:-1])
        f.close()
    
    

    # make a list of all the words including duplicates from every documents
    word_list = []
    for i in range(len(data_matrix)):
        word_list = word_list + data_matrix['desc'][i]
    
    unique_words = sorted(set(word_list))
    
    # size of the corpus
    global N
    N=len(data_matrix) # 1399
    # create tf-idf matrix for documents
    

    columns = ['index_terms']
    for i in range(N+1):
        columns.append('doc_'+str(i+1))
    # initiate the tf matrix
    tf_matrix = pd.DataFrame(columns=columns)
    # fill up the index terms column using the unique words
    tf_matrix['index_terms']=unique_words
    # initialize with all zeros
    tf_matrix.fillna(0, inplace=True)
    # create terms as indices
    tf_matrix.set_index('index_terms', inplace=True)
    # fill tf_matrix with the frequency of each term for every docs
    count = 0
    for j in unique_words:
        count += 1
        for k in range(len(data_matrix)):
            content = data_matrix['desc'][k] 
            docid = data_matrix['doc_id'][k]
            f = content.count(j)
            if f!=0:
                tf_matrix['doc_'+str(docid)][j] = f
        print("done for word:",j, "at level", count)
    print("Weighted term frequency matrix completed.") 
    
    tf_matrix.to_csv('tf_matrix.txt','\t')
    
    for i in range(1,N+2):
        tf_matrix['doc_'+str(i)] = tf_matrix['doc_'+str(i)] / max(tf_matrix['doc_'+str(i)])
        
        
    tf = pd.read_csv('tf_matrix.txt',sep='\t')
    tf.set_index('index_terms', inplace=True)

    
    for i in range(1,N+2):
        tf['doc_'+str(i)] = tf['doc_'+str(i)] / max(tf['doc_'+str(i)])
        
    # append a column D at the end of tf
    sLength = len(tf.index)
    D = pd.Series(np.zeros(sLength))
    tf = tf.assign(D=D.values)
    # append an IDF column
    IDF = pd.Series(np.zeros(sLength))
    tf = tf.assign(IDF=IDF.values)
    # fill the nonzero count at D
    """
    computing term frequecny and inverce-document frequency
    for every index terms. two columns are being added for this purpose.
    """
    for indices in range(len(tf.index)):
        di = sum(tf.loc[tf.index[indices]]>0)
        # update the di values |di|=non zero entries of a term w.r.t a doc
        tf['D'][tf.index[indices]] = di
        # IDF computation
        tf['IDF'][tf.index[indices]] = math.log(N/di,2)
    
    """
    tf-idf computation phase. compute tf-idf for each entry of the data 
    """
    global A
    A = tf
    for i in range(len(tf.index)): 
        A.loc[A.index[i]][0:N+1] = A.loc[A.index[i]][0:N+1] * A.IDF[i]
        
    #query = "what similarity laws must be obeyed when constructing aeroelastic models of heated high speed aircraft"
    """
    if a user selects auto=0, manual search input is being processed until enter is pressed.
    if a user selects auto=1, query file is read and processed and out put is saved in scores_output.txt
    """
    if auto==0:
        while True:
            user_query = tokenize(input("Search query >> "))
            if user_query == []:
                sys.exit()
            user_query_processing(user_query)
    elif auto==1:
        ts = time.clock()
        default_query_processing()
        te = time.clock()
        print("Time required to generate the ranked output for the query from file:", te-ts, "seconds")
    else:
        print("Please choose the value of auto as 0 or 1 and start again!")
        
    total_time = time.clock() - start
    print("Program ended running after ", total_time/60, "minutes!")
