# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 16:30:24 2015

@author: Mathurin
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from __future__ import division

G = nx.read_edgelist('ca-GrQc.txt', comments='#', delimiter='\t', create_using=nx.Graph(),
                 nodetype=int, data=True, edgetype=None, encoding='utf−8')
                 
print "Graph is connected:", nx.is_connected(G)
print "Number of nodes:", nx.number_of_nodes(G)
print "Number of edges:", nx.number_of_edges(G)
print "Number of connected components:", nx.number_connected_components(G)

GCC = list(nx.connected_component_subgraphs(G))[0]
print "Fraction of nodes in GCC:", GCC.number_of_nodes()/G.number_of_nodes()
print "Fraction of edges in GCC:", GCC.number_of_edges()/G.number_of_edges()



degree_sequence = G.degree().values()
plt.ylabel("Frequency")
plt.xlabel("Degree")
plt.hist(degree_sequence, bins=50)
print "Min degree:", np.min(degree_sequence)
print "Max degree:", np.max(degree_sequence)
print "Mean degree:", np.mean(degree_sequence)
print "Median degree:", np.median(degree_sequence)
print np.std(degree_sequence)

y=nx.degree_histogram(G)
plt.figure(1)
plt.plot( y , 'b-' , marker = 'o' ) 
plt.ylabel("Frequency")
plt.xlabel("Degree")
plt.show()

plt.figure(2)
plt.loglog( y , 'b-' , marker = 'o' ) 
plt.ylabel("Frequency")
plt.xlabel("Degree")
plt.show()

t = nx.triangles(G)
print "Number of triangles:", sum(t.values())/3

t_values = sorted(set(t.values()))
t_hist = [t.values().count(x) for x in t_values]

plt.figure(3)
plt.plot(t_hist, 'b-', marker='o')
plt.ylabel("")
plt.xlabel("")
plt.show()

plt.figure(4)
plt.loglog(t_hist, 'b-', marker='o')
plt.ylabel("")
plt.xlabel("")
plt.show()

print "Average clustering coefficients:", nx.average_clustering(G)

# Degree centrality
deg_centrality = nx.degree_centrality(G)
# Eigenvector centrality
eig_centrality = nx.eigenvector_centrality(G)


#Sort centrality values
sorted_deg_centrality = sorted(deg_centrality.items()) 
sorted_eig_centrality = sorted(eig_centrality.items())
# Extract centralities
deg_data=[b for a,b in sorted_deg_centrality] 
eig_data=[b for a,b in sorted_eig_centrality]

from scipy.stats.stats import pearsonr
print "Pearson correlation coefficient", pearsonr(deg_data, eig_data)

plt.figure(5)
plt.scatter(deg_data, eig_data)
plt.show()



R = nx.fast_gnp_random_graph(200, 0.1)
print "Graph is connected:", nx.is_connected(R)
print "Number of nodes:", nx.number_of_nodes(R)
print "Number of edges:", nx.number_of_edges(R)
print "Number of connected components:", nx.number_connected_components(R)

degree_sequence_R = R.degree().values()
plt.ylabel("Frequency")
plt.xlabel("Degree")
plt.hist(degree_sequence_R, bins=50)
print "Min degree:", np.min(degree_sequence_R)
print "Max degree:", np.max(degree_sequence_R)
print "Mean degree:", np.mean(degree_sequence_R)
print "Median degree:", np.median(degree_sequence_R)
print np.std(degree_sequence)

y_R=nx.degree_histogram(R)
plt.figure(6)
plt.plot( y_R , 'b-' , marker = 'o' ) 
plt.ylabel("Frequency")
plt.xlabel("Degree")
plt.show()



t_R = nx.triangles(R)
print "Number of trianglesfor the random graph:", sum(t_R.values())/3

t_values_R = sorted(set(t_R.values()))
t_hist_R = [t_R.values().count(x) for x in t_values_R]

plt.figure(7)
plt.plot(t_hist_R, 'b-', marker='o')
plt.ylabel("")
plt.xlabel("")
plt.show()

print "Average clustering coefficients for the random graph:", nx.average_clustering(R)

