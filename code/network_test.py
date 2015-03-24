# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 14:15:03 2015

@author: marcus
"""

import numpy as np
import numpy.random as rng
import networkx as nx
import pylab as pl
import sys, argparse
from givers import Giver, World
np.set_printoptions(precision=2)


def display_pair_sequence(playerA, playerB, outfile = 'apair.png'):
    # Display the hell out of it
    pl.clf()
    pl.subplot(3,1,1)
    pl.plot(playerA.utility_trail,'s-k')
    pl.plot(playerB.utility_trail,'o-k', markerfacecolor='white', alpha=.5)
    pl.title('utilities over time')
    biplayerA = np.max(playerA.utility_trail)
    biplayerB = np.max(playerB.utility_trail)
    biggest = max(biplayerA, biplayerB)
    #pl.gca().set_ylim(-0.5, biggest + .5)

    maxcount = max(np.max([playerA.counts_trail[x] for x in world.commodities]),np.max([playerA.counts_trail[x] for x in world.commodities]))
    pl.subplot(3,1,2)
    for x in world.commodities:
        pl.plot(playerA.counts_trail[x],'-s', alpha=.5, label = x)
    pl.gca().set_ylim(-0.5,maxcount+0.5)
    pl.ylabel(playerA.name)
    pl.gca().set_xticks([])
    l = pl.legend()

    pl.subplot(3,1,3)
    for x in world.commodities:
        pl.plot(playerB.counts_trail[x],'-o', alpha=.5, label = x)
    pl.gca().set_ylim(-0.5,maxcount+0.5)
    pl.ylabel(playerB.name)
    pl.gca().set_xticks([])
    pl.savefig(outfile,dpi=200)
    print 'wrote %s' % (outfile)

    
def display_utilities_heatmaps(wlim, util1, util2, args, outfile = 'utilities.png'):
    fig = pl.figure()
    z_min = 8.  #np.min(displayUtil.ravel())
    z_max = -8. #np.max(displayUtil.ravel())
    z_lim = max(abs(z_min), abs(z_max)) # just trying to get white to be zero!!

    pl.subplot(221) # top left is Agent 1's utility, for various Agent 2 strategies
    im = pl.imshow(util1.transpose(), interpolation='nearest', origin='lower', cmap=pl.cm.Spectral, extent=(-wlim,wlim,-wlim,wlim), vmin=-z_lim, vmax=z_lim)
    pl.gca().set_title('utility of playerA')
    pl.plot(playerA.W[1],playerA.W[2],'ok', markersize=10)
    #pl.gca().set_ylabel('w2 of agent 2')
    #pl.gca().set_xlabel('w1 of agent 2')
    pl.gca().text(0, -wlim*4/5, 'w1 of agent 2', ha='center', va='top', rotation=0)
    pl.gca().text(-wlim*4/5, 0, 'w2 of agent 2', ha='right', va='center', rotation=90)
    #CBI = pl.colorbar(im, orientation='vertical', shrink=0.8, extend='both')

    pl.subplot(222) # top right is Agent 2's utility, for various Agent 2 strategies
    im = pl.imshow(util2.transpose(), interpolation='nearest', origin='lower', cmap=pl.cm.Spectral, extent=(-wlim,wlim,-wlim,wlim), vmin=-z_lim, vmax=z_lim)
    pl.plot(playerA.W[1],playerA.W[2],'ok', markersize=10)
    pl.gca().set_title('utility of playerB')
    CBI = pl.colorbar(im, orientation='vertical', shrink=0.8, extend='both')

    pl.subplot(223) # the relative advantage of Agent 1 over Agent 2.
    im = pl.imshow((util1-util2).transpose(), interpolation='nearest', origin='lower', cmap=pl.cm.Spectral, extent=(-wlim,wlim,-wlim,wlim), vmin=-z_lim, vmax=z_lim)
    pl.plot(playerA.W[1],playerA.W[2],'ok', markersize=10)
    pl.gca().set_title('playerA - playerB')

    pl.subplot(224)
    pl.gca().axis('off')
    pl.gca().text(0,0,'playerA plays ' + str(playerA.get_weights()))
    CBI = pl.colorbar(im, orientation='vertical', shrink=0.8, extend='both')


    pl.savefig(outfile,dpi=200)
    print 'wrote %s' % (outfile)


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description="test out two donors")

    parser.add_argument("-t", "--num_steps", help="number of steps", type=int,
                         default=10)
    parser.add_argument("--verbose", help="use verbose output",
                        action="store_true")
    parser.add_argument('-A', '--Aweights', nargs='+', type=float, help='weights for agent B', required='True')
    parser.add_argument('-B','--Bweights', nargs='+', type=float, help='weights for agent B')


    world = World(commodities=['food', 'water'])
    playerA = Giver(world, 'Jim')
    playerB = Giver(world, 'Sue')
    playerA.add_neighbour(playerB)
    playerB.add_neighbour(playerA)
    world.add_node(playerA)
    world.add_node(playerB)


    args = parser.parse_args()
    print args.num_steps
    if args.verbose:
        print "verbosity turned on"

    playerA.set_weights(args.Aweights)
    SINGLE_TEST = False
    if args.Bweights:
        SINGLE_TEST = True
        playerB.set_weights(args.Bweights)


    # w0 = 10.0    # if +ve, agent never wants to PASS
    # w1 = 100.0  # if +ve, agent gives away least valuable first
    # w2 = 6.0      # if +ve, agent wants to give when other has given more.


    if SINGLE_TEST == True:   
        # Do a single run, with those two agents using the supplied weight values.
        playerA.set_counts( dict(zip(world.commodities, [10,1])))
        playerB.set_counts( dict(zip(world.commodities, [1,10])))
        run_two_givers(playerA, playerB, args.num_steps)
        playerA.display()
        playerB.display()
        display_pair_sequence(playerA, playerB, 'sequences.png')
        print np.mean(playerA.utility_trail), np.mean(playerB.utility_trail)

    if SINGLE_TEST == False:   
        # we will test all sorts of weights for the second player.
        print 'Running one against many alternatives and making a plot...'
        wlim = 20 # wlim sets limits on the weight space we will explore.
        n = 41
        playerBw1 = np.linspace(-wlim, wlim, n)
        playerBw2 = np.linspace(-wlim, wlim, n)
        #X,Y = np.meshgrid(playerBw1,playerBw2)
        finalUtil_1 = np.zeros(shape=(n,n))
        finalUtil_2 = np.zeros(shape=(n,n))
        for i1,val1 in enumerate(playerBw1):
            for i2,val2 in enumerate(playerBw2):
                
                #playerA.set_weights([playerA.W[0], val1, val2])
                playerB.set_weights([playerA.W[0], val1, val2])  # NOTE: uses same w0 as Agent 1.
                # Reset the initial quantities to be unequal.
                playerA.set_counts( dict(zip(world.commodities, [10,1])))
                playerB.set_counts( dict(zip(world.commodities, [1,10])))

                run_two_givers(playerA, playerB, args.num_steps)
                finalUtil_1[i1,i2] = np.mean(playerA.utility_trail)
                finalUtil_2[i1,i2] = np.mean(playerB.utility_trail)
                if (val1 == playerA.W[1]) and (val2 == playerA.W[2]):
                    print ' Hey!! ', playerA.utility_trail[-1], playerB.utility_trail[-1]
        # show the results as heat-maps or contours
        display_utilities_heatmaps(wlim, finalUtil_1, finalUtil_2, args, 'utilities_%d.png' % (args.num_steps))

        playerA.display()
        playerB.display()
    
#------------------------------------------------------

