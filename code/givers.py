# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 10:42:05 2015

@author: marcus
Trading until the cows come home.
Marcus Frean.
"""
import numpy as np
import numpy.random as rng
import pylab as pl
from network_pic import *

commodities = ['apples','shoes','wheat']        
      
#%% ----------- start of Trader class definition ------------------------------------------
class Trader:
    def __init__(self, name = ''):
        self.name = name
        self.count =  {x: 1+rng.randint(4)  for x in commodities}
        vals = rng.dirichlet(3.0*np.ones(len(commodities)))
        # I like this as I can do inheritance of "similar" values
        # easily: use scaled up "parent" vals as the alpha!
        # self.valuation = {x: vals[i]  for i,x in enumerate(commodities)}
        self.neighbours = []
        # keys will be the neighbours for the following dictionaries
        self.given_val = {}       
        self.recd_val = {}
        self.num_trades = {} 

    def __str__(self):
        return self.name

    def _get_utility(self, suggested_counts):
        """
        return the utility to this agent of having the suggested commodity counts
        """
        return np.sum(np.log(1+suggested_counts))

    def _get_utility_increase(self, commod, increment):
        """
        return the utility INCREASE to this agent of having its commodity counts change by the suggested increments/decrements. 
        """
        changes = np.zeros(len(commodities))
        changes[comm
        return _get_utility(self.counts + changes) - _get_utility(self.counts)

    def add_neighbour(self, new_neighbour):
        if (new_neighbour in self.neighbours) or (new_neighbour.name == self.name):
            return False
        else:
            self.neighbours.append(new_neighbour)
            self.given_val[new_neighbour] = 0.0
            self.recd_val[new_neighbour]  = 0.0
            self.num_trades[new_neighbour] = 0
            return True

    def set_count(self, commodity, i): 
        self.count[commodity] = i

    def give(self, commodity, nb): 
        # check you can give it first!...
        if self.count[commodity] == 0: return False
        # evaluate the gift, and add the valuation to self.given_val[nb]
        
        value_lost = self._get_utility_increase(self, commodity, -1)
        print commodity, self.valuation[commodity], nb
        self.given_val[nb] += self.valuation[commodity]
        nb.recd_val[self] += nb.valuation[commodity]
        # you gave it away, so decrement the count...
        self.count[commodity] -= 1
        nb.count[commodity] += 1
        return True
        
    def receive(self, commodity, nb): 
        # evaluate the gift, and add the valuation to self.recd_val[nb]
        self.given_val[nb] += self.valuation[commodity]
        self.count[commodity] += 1


    def display(self):
        print '--------------------------------------------'
        print 'Trader ', self.name
        for nb in self.neighbours:
            print 'On %4d trades: gave %.1f, recd %.1f \t to %s' % (self.num_trades[nb], self.given_val[nb], self.recd_val[nb], nb)
        for x in commodities:
            print '%8s: %3d' % (x, self.count[x])

    def do_one_gift():
        """Consider giving one of a commodity to one neighbour. Evaluate
        willingness to do this for all commodities and all neighbours,
        including a do-nothing option.

        Then DO IT, ie. update count, given_val, recd_val, num_trades
        for both self, and the receiver.

        """
        best_will, best_neighbour, best_commodity = -1000000000.0, None, None
        for nb in self.neighbours:
            for c in commodities:
                # eval the willingness to give this c to this nb.
                willingness = 1 ### WORK HERE WHEN YOU WAKE UP!!!

# ----------- end of Trader class definition ------------------------------------------

#%% Other methods - the global stuff

def do_one_season():
    """Zillions of trades happen. Agents accumulate rewards. 

    Should we have a fixed number of trades? Should rewards be grand sum,
    or on-the-fly exponentially weighted moving average?
    """
    for t in range(3):  # will be much bigger!
        # pick a random trader
        tr = traders[rng.randint(len(traders))]
        tr.do_one_gift()


def adapt_all_traders():
    """Everyone reconsiders their position, given their relative
    performance.  For example, perhaps everyone adopts the values and
    strategy (?) of someone else who is doing better. BUT HEY that
    won't do: if I'm in an apple-rich region do I suddenly want to
    value shoes? Sure, IF I DO then I might as well grab the
    shoe-guy's behavioural strategy too. But this seems to defy
    locality.

    """


if __name__ == '__main__':

    # Define the trader network
    names = ['Jim', 'Yoyo', 'Bianca', 'Bob', 'Serena', 'Kurt', 'Shaun']
    traders = []
    for name in names:
        tr = Trader(name)
        traders.append(tr)
    # link them up somehow...
    edge_ends_list = []
    edge_labels = []
    for tr1 in traders:
        n, attempts = 0, 0
        while (n<1) and (attempts<10):
            attempts = attempts + 1
            tr2 = traders[rng.randint(len(traders))]
            isNewEdge = (tr1.add_neighbour(tr2) and tr2.add_neighbour(tr1))
            if isNewEdge:
                edge_ends_list.append((tr1.name, tr2.name))
                edge_labels.append(tr1.name + ' and ' + tr2.name)
                n = n + 1
    traders[0].give('wheat', traders[1])  # test the gifting method.

    for tr in traders:    
        tr.display()

    #%% Show the whole network as a picture    
    G, node_posns = generate_drawable_graph(names, edge_ends_list)
    draw_graph(G, node_posns, names, edge_ends_list, edge_labels)
