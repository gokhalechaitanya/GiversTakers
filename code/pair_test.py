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
import pylab as pl

def run_two_givers(g1, g2, num_gifts, dynamics='alt'):

    if dynamics == 'alt':
                    
        for t in range(num_gifts):
            # g1 does a gifting!
            g1.do_one_gift(temperature=1.0, verbose=False)
            g1.append_trail()
            g2.append_trail()
            # g2 does a gifting!
            g2.do_one_gift(temperature=1.0, verbose=False)
            g2.append_trail()
            g1.append_trail()
            
    else: 
        print 'Do not know what dynamics to use'
    
    
import sys
if __name__ == '__main__':
    print sys.argv
    w0 = -10.0 # -ve means never wants to PASS
    w1 = 20.0   # +ve means gives away least valuable first
    w2 = -6.0   # wants to be "in debt".
    if len(sys.argv) > 1:
        w0 = float(sys.argv[1])
        w1 = float(sys.argv[2])
        w2 = float(sys.argv[3])
    print 'Using these weights for both parties: ', w0, w1, w2

    world = World(commodities=['love', 'respect', 'backrub'])

    g1 = Giver(world, 'Jim', start_counts={'love':40, 'respect':20, 'backrub':0})
    g2 = Giver(world, 'Bob', start_counts={'love':0, 'respect':20, 'backrub':40})

    g1.add_neighbour(g2)
    g2.add_neighbour(g1)
    world.add_node(g1)
    world.add_node(g2)

    g1.set_weights(w0, w1, w2)
    g2.set_weights(w0, w1, w2)
    
    print 'Running some gifting here'
    run_two_givers(g1, g2, 12)

    
    # Display the hell out of it
    pl.clf()
    pl.subplot(3,1,1)
    pl.plot(g1.utility_trail,'s-k', g2.utility_trail,'o-k')
    pl.title('utilities over time')
    big1 = np.max(g1.utility_trail)
    big2 = np.max(g2.utility_trail)
    biggest = max(big1, big2)
    pl.gca().set_ylim(-0.5, biggest + .5)
    pl.subplot(3,1,2)
    for x in world.commodities:
        pl.plot(g1.counts_trail[x],'-s', alpha=.5, label = x)
    pl.gca().set_ylim(-0.5,40)
    pl.xlabel(g1.name)
    pl.gca().set_xticks([])
    l = pl.legend()
    pl.subplot(3,1,3)
    for x in world.commodities:
        pl.plot(g2.counts_trail[x],'-o', alpha=.5, label = x)
    pl.gca().set_ylim(-0.5,40)
    pl.xlabel(g2.name)
    pl.gca().set_xticks([])
    pl.savefig('apair.png')
