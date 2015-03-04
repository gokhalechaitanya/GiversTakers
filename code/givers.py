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
np.set_printoptions(precision=2)

commodities = ['apples','shoes','wheat']        
      
#%% ----------- start of Trader class definition ------------------------------------------
class Trader:
    def __init__(self, name = ''):
        self.name = name
        #self.count =  rng.randint(1,10, size=len(commodities))
        #vals = rng.dirichlet(3.0*np.ones(len(commodities)))
        # I like this as I can do inheritance of "similar" values
        # easily: use scaled up "parent" vals as the alpha!
        self.count = {x: rng.randint(0,5)  for x in commodities}
        self.neighbours = []
        # keys for the following will be the neighbours for the following dictionaries
        self.given_val = {}       
        self.recd_val = {}
        self.num_trades = {} 
        self.W0 = rng.normal()  
        # strength of tendency to PASS (avoid gifting at all)

        self.W1 = rng.normal()  
        # strength of tendency to base trades on the value being given away
        # (a +ve W1 means agent prefers low-loss trades)

        self.W2 = rng.normal()  
        # strength of tendency to base trades on shared history of trades
        # (a +ve W2 means agent prefers to trade when it is "in the red")

    def __str__(self):
        return self.name

    def _get_utility(self):
        """
        return the utility to this agent of having the suggested commodity counts
        """
        util = 0.0
        for c in commodities:
            util += np.log(1 + self.count[c])  # for example....
        return util 

                
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


    def display(self):
        print '--------------------------------------------'
        print 'Trader ', self.name
        print 'W0,W1,W2 : %.2f, %.2f, %.2f' %(self.W0, self.W1, self.W2)
        for nb in self.neighbours:
            print 'On %4d trades: gave %.2f, recd %.2f \t with %s' % (self.num_trades[nb], self.given_val[nb], self.recd_val[nb], nb)
        for x in commodities:
            print '%8s: %3d' % (x, self.count[x])
        print 'utility = %.2f' %(self._get_utility())
        
    def do_one_gift(self, verbose=False):
        """Consider giving one of a commodity to one neighbour. Evaluate
        willingness to do this for all commodities and all neighbours,
        including a do-nothing option.

        Then DO IT (with the give() method)

        """
        if verbose: self.display()
        
        #%% first we decide what to do
        drive = np.zeros(shape=(len(self.neighbours), len(commodities)), dtype=float)
        base_utility = self._get_utility()   
        for j,nb in enumerate(self.neighbours):
            for i,c in enumerate(commodities):
                # eval the willingness to give this c to this nb.
                # Thought experiment: "what if I gave c away to nb?"
                if self.count[c] > 0:
                    self.count[c] -= 1 
                    loss = self._get_utility() - base_utility
                    self.count[c] += 1  # cos it's only a thought expt! 
                else:
                    loss = -np.Inf
                drive[j,i] = self.W1 * loss + self.W2 * (self.recd_val[nb] - self.given_val[nb])


        willingness = np.exp(drive)
        total_willing = np.sum(np.ravel(willingness))
        total_unwilling = np.exp(self.W0)
        # First decision: do nothing?
        if total_willing > total_unwilling:
            willingness = willingness / np.sum(np.ravel(willingness))
            if verbose: 
                print '#### %s considers gifting ' % (self.name), commodities
                for j,nb in enumerate(self.neighbours):
                    print '#### to %5s: '%(nb), willingness[j]
            best_nb_index, best_commod_index = np.unravel_index(willingness.argmax(), willingness.shape)
            best_nb = self.neighbours[best_nb_index]
            best_commod = commodities[best_commod_index]
            if verbose: print '#### Best option: give %s to %s' % (best_commod, best_nb)
                
            #%% So do it! Then DO IT, ie. update count, given_val, recd_val, num_trades for both self, and the receiver.
            # Q. how to decide whether to just do nothing though?
            if verbose: best_nb.display()
            self.give(best_commod, best_nb)
            if verbose: 
                self.display()
                best_nb.display()
        else:  
            # Agent prefers to PASS regarding gifting, this round
            if verbose: print '#### best to PASS: %.2f better than %.2f' %(total_unwilling, total_willing)

    def give(self, commodity, trader):
        """ Give a specific commodity to a specific trader """
        if self.count[commodity] == 0:
            return
            
        base_utility = self._get_utility()   
        self.count[commodity]    -= 1
        lost = self._get_utility() - base_utility
        self.given_val[trader] -= lost
        self.num_trades[trader] += 1
    
        base_utility = trader._get_utility()   
        trader.count[commodity] += 1
        gained = trader._get_utility() - base_utility
        trader.recd_val[self] += gained
        trader.num_trades[self] += 1
        
# ----------- end of Trader class definition ------------------------------------------

#%% Other methods - the global stuff

def do_one_season(traders, verbose=False):
    """Zillions of trades happen. Agents accumulate rewards. 

    Should we have a fixed number of trades? Should rewards be grand sum,
    or on-the-fly exponentially weighted moving average?
    """
    for t in range(10):  # will be much bigger!
        # do one round of all traders, in a random order
        for i in rng.permutation(len(traders)):
            tr = traders[i]
            tr.do_one_gift(verbose)


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
    names = ['Bulbulia', 'Yoyo'] #, 'Bianca', 'Bob', 'Serena', 'Kurt', 'Shaun', 'Shona', 'Grant', 'Bill', 'Shorty', 'Kim', 'Tyler']
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

    # Test one trade:
    #tr = traders[rng.randint(len(traders))]
    #tr.do_one_gift()

    for tr in traders: tr.display()
    print '^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^'
    do_one_season(traders, verbose=False)
    for tr in traders: tr.display()
    

    #%% Show the whole network as a picture    
    G, node_posns = generate_drawable_graph(names, edge_ends_list)
    edge_thck = 6
    #edge_thck = []
    #for (name1,name2) in edge_ends_list:
    #    edge_thck.append(traders[name1].num_trades[traders[name2]])
    draw_graph(G, node_posns, names, edge_ends_list, edge_labels, edge_thck)
