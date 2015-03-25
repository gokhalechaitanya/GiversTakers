# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 11:17:57 2015

@author: marcus
"""

import networkx as nx
import matplotlib.pyplot as plt
import types

def generate_drawable_graph(node_names, edge_ends_list):
    """ Probably call this once only: else the nodes may jump from
    place to place each time you display the same graph!
    """
    # create networkx graph
    G=nx.Graph()  #alt:    G=nx.MultiDiGraph()
    # add nodes
    for node in node_names:
        G.add_node(node)
    # add edges
    for e in edge_ends_list:
        G.add_edge(e[0], e[1])
    # Choose one fo the following:
    #node_posns=nx.shell_layout(G)
    node_posns=nx.spring_layout(G)
    #node_posns=nx.spectral_layout(G)
    #node_posns=nx.random_layout(G)
    return G, node_posns
    

def draw_graph(G, node_posns, node_labels_list, node_sizes, edge_ends_list, edge_labels=None, edge_thck=None, filename=None):
    """ Assumes you've already called generate_drawable_graph() to get G and node_posns.
    """
    plt.clf()
    if type(node_sizes) is not list:
        node_sizes = 1500
    if type(edge_thck) is not list:
        edge_thck = [6] * len(edge_ends_list)

    edge_labels_dict = dict(zip(edge_ends_list, edge_labels))
    nx.draw_networkx_nodes(G, node_posns, node_labels_list, node_size=node_sizes, alpha=0.5, node_color='greenyellow')
    nx.draw_networkx_labels(G, node_posns, font_size=12, font_family='sans-serif')

    newlist_edge_thck = []
    for (a,b) in G.edges():
        if (a,b) in edge_ends_list:
            #print 'yes! ', (a,b), ' seems to be item ', edge_ends_list.index((a,b)), ' with thck ', edge_thck[edge_ends_list.index((a,b))]
            newlist_edge_thck.append(edge_thck[edge_ends_list.index((a,b))])
        elif (b,a) in edge_ends_list:
            #print 'yes! ', (b,a), ' seems to be item ', edge_ends_list.index((b,a)), ' with thck ', edge_thck[edge_ends_list.index((b,a))]
            newlist_edge_thck.append(edge_thck[edge_ends_list.index((b,a))])
        else:
            print 'ooops, seems ',(a,b), ' is not in ', edge_ends_list

    nx.draw_networkx_edges(G, node_posns, width=newlist_edge_thck, alpha=0.25,edge_color='blue')
    #nx.draw_networkx_edge_labels(G, node_posns, edge_labels=edge_labels_dict, label_pos=0.4, font_size=9)

    #nx.draw(G, pos)
    plt.axis('off')
    #plt.show()
    if filename != None:
        plt.draw()
        plt.savefig(filename, dpi=200)
        print('Wrote %s' % (filename))
    

if __name__ == '__main__':
    num_nodes = 10
    node_names = map(chr, range(65, 65+num_nodes))
    print 'node_names are ', node_names
    node_list = []
    node_sizes_list = []
    edge_ends_list = []
    edge_thck = []
    edge_labels = []
    tmpsize = 200

    for i in range(len(node_names)):
        node_sizes_list.append(tmpsize)
        edge_thck.append(i+1)
        tmpsize += 200
    for i in range(len(node_names)-1):
        edge_ends_list.append((node_names[i],node_names[i+1]))
        edge_labels.append(node_names[i] + ' and ' + node_names[i+1])
    
    G, node_posns = generate_drawable_graph(node_names, edge_ends_list)
    draw_graph(G, node_posns, node_names, node_sizes_list, edge_ends_list, edge_labels, edge_thck, 'network_pic_eg.png')

