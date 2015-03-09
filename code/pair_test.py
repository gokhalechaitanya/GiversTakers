# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 14:15:03 2015

@author: marcus
"""

import numpy as np
import numpy.random as rng
import networkx as nx
np.set_printoptions(precision=2)
from givers import Giver, World


def run_two_givers(gvr1, gvr2, num_gifts, dynamics='alt'):
    if dynamics == 'alt':
        # pick a giver at random
        if rng.random() > 0.5:
            g1, g2 = gvr1, gvr2
        else:
            g1, g2 = gvr2, gvr1
            
        # g1 does a gifting!
        for t in range(num_gifts):
            g1.do_one_gift(verbose=False)
            g2.do_one_gift(verbose=False)
        
    else: 
        print 'Do not know what dynamics to use'
    
    
    
    
if __name__ == '__main__':
    
    world = World(commodities=['love', 'respect', 'backrub'])
    
    g1 = Giver(world, 'Jim') #, np.array([1,1,1]))
    g2 = Giver(world, 'Bob')
    g1.add_neighbour(g2)
    g2.add_neighbour(g1)
    world.add_node(g1)
    world.add_node(g2)
    
    w0, w1, w2 = 0.0, 1.0, 5.0
    g1.set_weights(w0, w1, w2)
    g2.set_weights(w0, w1, w2)
    
    g1.display()
    g2.display()
    print 'Running some gifting here'
    run_two_givers(g1, g2, 30)
    g1.display()
    g2.display()
        