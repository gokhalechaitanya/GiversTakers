# -*- coding: utf-8 -*-
"""
@author: marcus
"""

import numpy as np
import numpy.random as rng
import networkx as nx
import pylab as pl
import sys, argparse
from givers import *  #Giver, World
np.set_printoptions(precision=2)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="test out a network of givers")

    parser.add_argument("-t", "--num_steps", help="number of steps", type=int,
                         default=10)
    parser.add_argument("-v", "--verbose", help="use verbose output",
                        action="store_true")
    parser.add_argument('-A', '--Aweights', nargs='+', type=float, help='weights for agent B', required='True')
    parser.add_argument('-B','--Bweights', nargs='+', type=float, help='weights for agent B')

    args = parser.parse_args()
    print args.num_steps
    if args.verbose:
        print "verbosity turned on"

    world = World(commodities=['food', 'water', 'shelter'])
    players = []
    Q = 24 # the initial total number of items each individual possesses.
    quantities = [0] * len(world.commodities)
    players.append(Giver(world, 'Jim', dict(zip(world.commodities, quantities))))
    players[-1].count[world.commodities[0]] = Q
    players[-1].set_weights(args.Aweights)
    players.append(Giver(world, 'Joe', dict(zip(world.commodities, quantities))))
    players[-1].count[world.commodities[1]] = Q
    players[-1].set_weights(args.Aweights)
    players.append(Giver(world, 'Jaz', dict(zip(world.commodities, quantities))))
    players[-1].count[world.commodities[2]] = Q
    players[-1].set_weights(args.Aweights)
    links = [[0,1],[0,2],[1,2]]

    if args.Bweights:
        players.append(Giver(world, 'Sue', dict(zip(world.commodities, quantities))))
        players[-1].count[world.commodities[0]] = Q
        players[-1].set_weights(args.Bweights)
        players.append(Giver(world, 'Sam', dict(zip(world.commodities, quantities))))
        players[-1].count[world.commodities[1]] = Q
        players[-1].set_weights(args.Bweights)
        players.append(Giver(world, 'Syl', dict(zip(world.commodities, quantities))))
        players[-1].count[world.commodities[2]] = Q
        players[-1].set_weights(args.Bweights)
        links = [[0,1],[0,2],[1,2],[1,3],[2,4],[3,4],[3,5],[4,5]]

    for link in links:
        players[link[0]].add_neighbour(players[link[1]])
        players[link[1]].add_neighbour(players[link[0]]) # the reciprocal link
    for player in players:
        world.add_node(player)

    print '&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&'
    world.do_one_season(args.num_steps, verbose=args.verbose)
    print '&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&'

    for player in players:
        player.display()
    #world.display_pair_sequence(players[0], players[1], 'network_test_example_seq.png')
    world.display_all_sequences('network_test_all_seq.png')
    world.show_network('network_test_result.png')



