# -*- coding: utf-8 -*-
"""
@author: marcus
"""

import numpy as np
import numpy.random as rng
import networkx as nx
import pylab as pl
import sys, argparse
from givers import Giver, World
np.set_printoptions(precision=2)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="test out a network of givers")

    parser.add_argument("-t", "--num_steps", help="number of steps", type=int,
                         default=10)
    parser.add_argument("--verbose", help="use verbose output",
                        action="store_true")
    parser.add_argument('-A', '--Aweights', nargs='+', type=float, help='weights for agent B', required='True')
    parser.add_argument('-B','--Bweights', nargs='+', type=float, help='weights for agent B')


    world = World(commodities=['food', 'water'])
    players = []
    players.append(Giver(world, 'Jim'))
    players.append(Giver(world, 'Joe'))
    players.append(Giver(world, 'Jap'))
    players.append(Giver(world, 'Sue'))
    players.append(Giver(world, 'Sly'))
    players.append(Giver(world, 'Sam'))

    links = [[0,1],[0,2],[1,2],[1,3],[2,4],[3,4],[3,5],[4,5]]

    for link in links:
        players[link[0]].add_neighbour(players[link[1]])
        players[link[1]].add_neighbour(players[link[0]]) # the reciprocal link
    for player in players:
        world.add_node(player)


    args = parser.parse_args()
    print args.num_steps
    if args.verbose:
        print "verbosity turned on"

    for player in players[:3]:
        player.set_weights(args.Aweights)
        print player.name, player.get_weights()
    for player in players[3:]:
        player.set_weights(args.Bweights)
        print player.name, player.get_weights()

    # Do a single run
    for player in players:
        if rng.random() < 0.5:
            player.set_counts( dict(zip(world.commodities, [10,1])))
        else:
            player.set_counts( dict(zip(world.commodities, [1,10])))

    world.do_one_season(args.num_steps, verbose=args.verbose)
    for player in players:
        player.display()
    #display_sequences(playerA, playerB, 'sequences.png')
    world.show_network()



