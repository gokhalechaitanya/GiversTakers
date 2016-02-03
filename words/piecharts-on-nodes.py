# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 11:46:11 2015

@author: marcus
"""


import networkx as nx
import matplotlib.pyplot as plt
G=nx.complete_graph(4)
print G


#%%
pos=nx.spring_layout(G)
print pos

#%%
fig=plt.figure(figsize=(5,5))
ax=plt.axes([0,0,1,1])
ax.set_aspect('equal')
nx.draw_networkx_edges(G,pos,ax=ax)

plt.xlim(-0.5,1.5)
plt.ylim(-0.5,1.5)

trans=ax.transData.transform
trans2=fig.transFigure.inverted().transform

piesize=0.2
p2=piesize/2.0
for n in G:
    xx,yy=trans(pos[n]) # figure coordinates
    xa,ya=trans2((xx,yy)) # axes coordinates
    a = plt.axes([xa-p2,ya-p2, piesize, piesize])
    a.set_aspect('equal')
    fracs = [15,30,45, 10]
    a.pie(fracs)

plt.savefig('pc.png')
