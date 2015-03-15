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
import sys

def set_init_quantities(ag1, ag2):
    ag1.set_count('love', 20)  
    ag1.set_count('respect', 10)        
    ag1.set_count('backrub', 0)        
    ag2.set_count('love', 0)        
    ag2.set_count('respect', 10)        
    ag2.set_count('backrub', 20)        


def run_two_givers(g1, g2, num_gifts, dynamics='alt'):

    if dynamics == 'alt':
        g1.utility_trail = []
        g2.utility_trail = []
        
        for t in range(num_gifts):
            # g1 does a gifting!
            g1.do_one_gift(temperature=1.0, verbose=False)
            g1.append_trail()
            g2.append_trail()
            # g2 does a gifting!
            g2.do_one_gift(temperature=1.0, verbose=False)
            g2.append_trail()
            g1.append_trail()
        return g1.utility_trail[-1], g2.utility_trail[-1]
    else: 
        sys.exit('Do not know what dynamics to use')

        
def display_pair_sequence(g1, g2, outfile = 'apair.png'):
    # Display the hell out of it
    pl.clf()
    pl.subplot(3,1,1)
    pl.plot(g1.utility_trail,'s-k')
    pl.plot(g2.utility_trail,'o-k', markerfacecolor='white')
    pl.title('utilities over time')
    big1 = np.max(g1.utility_trail)
    big2 = np.max(g2.utility_trail)
    biggest = max(big1, big2)
    pl.gca().set_ylim(-0.5, biggest + .5)
    pl.subplot(3,1,2)
    for x in world.commodities:
        pl.plot(g1.counts_trail[x],'-s', alpha=.5, label = x)
    pl.gca().set_ylim(-0.5,20)
    pl.ylabel(g1.name)
    pl.gca().set_xticks([])
    l = pl.legend()
    pl.subplot(3,1,3)
    for x in world.commodities:
        pl.plot(g2.counts_trail[x],'-o', alpha=.5, label = x)
    pl.gca().set_ylim(-0.5,20)
    pl.ylabel(g2.name)
    pl.gca().set_xticks([])
    pl.savefig(outfile,dpi=200)
    print 'wrote %s' % (outfile)

    
def display_utilities_heatmap(displayUtil, outfile = 'utilities.png'):
    fig = pl.figure()   
    z_min = np.min(displayUtil.ravel())
    z_max = np.max(displayUtil.ravel())
    z_lim = max(abs(z_min), abs(z_max)) # just trying to get white to be zero!!
    im = pl.imshow(displayUtil, interpolation='nearest', origin='lower', cmap=pl.cm.Spectral, extent=(-5,5,-5,5), vmin=-z_lim, vmax=z_lim)
    pl.gca().set_title('rel adv util2 vs (w0=%.1f, w1=%.1f, w2=%.1f)' % (g1w0, g1w1, g1w2))
    pl.gca().set_ylabel('w1 of agent 2')
    pl.gca().set_xlabel('w2 of agent 2')
    levels = np.arange(np.min(displayUtil), np.max(displayUtil), 0.5)
    #CS = pl.contour(displayUtil, levels, origin='lower',
    #                         linewidths=2, extent=(-5,5,-5,5))
    #CB = pl.colorbar(CS, shrink=0.8, extend='both') #colorbar for the contours
    # We can still add a colorbar for the image, too.
    CBI = pl.colorbar(im, orientation='vertical', shrink=0.8, extend='both')

    pl.savefig(outfile,dpi=200)
    print 'wrote %s' % (outfile)


if __name__ == '__main__':

    world = World(commodities=['love', 'respect', 'backrub'])
    g1 = Giver(world, 'Jim')
    g2 = Giver(world, 'Bob')
    g1.add_neighbour(g2)
    g2.add_neighbour(g1)
    world.add_node(g1)
    world.add_node(g2)

    """
    w0 = 10.0    # if +ve, agent never wants to PASS
    w1 = 100.0  # if +ve, agent gives away least valuable first
    w2 = 6.0      # if +ve, agent wants to give when other has given more.
    """
    print "num args is ", len(sys.argv)
    if len(sys.argv) < 4:
        sys.exit('usage: python %s  w0 w1 w2 for giver A  [w0 w1 w2 for giver B]' % (sys.argv[0]))
    elif len(sys.argv) >= 4:
        g1w0 = float(sys.argv[1])
        g1w1 = float(sys.argv[2])
        g1w2 = float(sys.argv[3])
        g1.set_weights(g1w0, g1w1, g1w2)
        g1.display()
    if len(sys.argv) == 7:
        SINGLE_TEST = True
        g2w0 = float(sys.argv[4])
        g2w1 = float(sys.argv[5])
        g2w2 = float(sys.argv[6])
        g2.set_weights(g2w0, g2w1, g2w2)
        g2.display()
    else:
        SINGLE_TEST = False

    
    if SINGLE_TEST == True:   
        # Do a single run, with those two agents using the supplied weight values.
        print 'Running some gifting here'
        set_init_quantities(g1, g2)
        final_util1, final_util_2 = run_two_givers(g1, g2, 120)
        print final_util1, final_util_2    
        display_pair_sequence(g1, g2, 'sequences.png')

    if SINGLE_TEST == False:   
        # we will test all sorts of weights for the second player.
        print 'Running one against many alternatives and making a plot...'
        g2w1 = np.linspace(-5., 5, 11)
        g2w2 = np.linspace(-5., 5, 11)
        X,Y = np.meshgrid(g2w1,g2w2)
        finalUtil_1 = np.zeros(shape=X.shape)
        finalUtil_2 = np.zeros(shape=X.shape)
        for i1,val1 in enumerate(g2w1):
            for i2,val2 in enumerate(g2w2):
                
                g2.set_weights(g1w0, val1, val2)  # NOTE: uses same w0 as Agent 1.
                # Reset the initial quantities to be unequal.
                set_init_quantities(g1, g2)
            
                # And set the "memories" to zero transactions.
                g1.blank_memories()
                g2.blank_memories()

                u1, u2 = run_two_givers(g1, g2, 120)
                finalUtil_1[i1,i2] = u1
                finalUtil_2[i1,i2] = u2

        # show the results as heat-maps or contours
        utilityDiff = finalUtil_2 - finalUtil_1
        display_utilities_heatmap(finalUtil_1, 'utilities.png')
    
#------------------------------------------------------

