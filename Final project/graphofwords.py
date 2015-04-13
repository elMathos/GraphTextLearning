# -*- coding: utf-8 -*-
"""
represent each document as a graph

@author: Filfeul
"""

from __future__ import division
import networkx as nx
import numpy as np

class GraphOfWords :
    def __init__ (self,train_data,dico,sizeWindow):
        dico=np.asarray(dico)
        self.train_data=train_data
        self.n_doc = len(train_data)
        self.n_words = len(dico)
        self.sizeWindow=sizeWindow
        self.documentTerm=np.zeros((self.n_doc,self.n_words))
        self.graphs=[]        
        for i in range (self.n_doc):
            adjacencyMat=np.zeros((self.n_words,self.n_words))
            doc = np.asarray(train_data[i])
            for j in range (len(doc)):
                index1=np.nonzero(dico==doc[j])
                for k in range (1,min(sizeWindow,len(doc)-j)):
                    index2=np.nonzero(dico==doc[j+k])
                    adjacencyMat[index1[0][0],index2[0][0]]+=1
            graph=nx.from_numpy_matrix(adjacencyMat)  
            self.graphs.append(graph)    
        
                    
    def compute_documentTerm(self,mode):
        if mode==0 :
            for i in range(self.n_doc) :
                deg_centrality = nx.degree_centrality(self.graphs[i])
                for j in range(self.n_words):
                    self.documentTerm[i,j]=deg_centrality[j]
        elif mode==1:
            for i in range(self.n_doc):
                    eig_centrality=nx.eigenvector_centrality(self.graphs[i])
                    for j in range(self.n_words):
                        self.documentTerm[i,j]=eig_centrality[j]