# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 10:42:05 2015

@author: marcus
Trading until the cows come home.
Marcus Frean.
"""
import numpy as np
import numpy.random as rng
import networkx as nx
from network_pic import *
np.set_printoptions(precision=2)



#%% ----------- start of Giver class definition ------------------------------------------
class Giver:
    def __init__(self, world, name, start_counts=None):
        self.world = world # the world this agent lives within.        
        self.name = name
        #self.count =  rng.randint(1,10, size=len(    self.world.commodities))
        #vals = rng.dirichlet(3.0*np.ones(len(self.world.commodities)))
        # I like this as I can do inheritance of "similar" values
        # easily: use scaled up "parent" vals as the alpha!
        if start_counts is None:
            self.count = {x: rng.randint(0,5)  for x in self.world.commodities}            
        else:
            self.count = start_counts        
            
        self.neighbours = []
        # keys for the following will be the neighbours for the following dictionaries
        self.given_val = {}       
        self.recd_val = {}
        self.W0 = rng.normal()  
        # strength of tendency to PASS (avoid gifting at all)

        self.W1 = rng.normal()  
        # strength of tendency to base trades on the value being given away
        # (a +ve W1 means agent prefers low-loss trades)

        self.W2 = rng.normal()  
        # strength of tendency to base trades on shared history of trades
        # (a +ve W2 means agent prefers to trade when it is "in the red")

        self.utility_trail = [] # just a record of utility over a season
        self.counts_trail = {x: []  for x in self.world.commodities}
        # that is just a record for each commodity over a season
        
        self.append_trail()

    def __str__(self):
        return self.name

    def get_utility(self):
        """
        return the utility to this agent of having the current commodity counts
        """
        util = 0.0
        for c in self.world.commodities:
            util += np.log(1 + self.count[c])  # for example....
        return util 

    def append_trail(self):
        self.utility_trail.append(self.get_utility())
        for x in self.world.commodities:
            self.counts_trail[x].append(self.count[x])
                
    def add_neighbour(self, new_neighbour):
        if (new_neighbour in self.neighbours) or (new_neighbour.name == self.name):
            return False
        else:
            self.neighbours.append(new_neighbour)
            self.given_val[new_neighbour] = 0.0
            self.recd_val[new_neighbour]  = 0.0
            return True

    def set_count(self, commodity, i): 
        self.count[commodity] = i

    def set_weights(self, w0, w1, w2): 
        self.W0 = w0
        self.W1 = w1
        self.W2 = w2

    def display(self):
        print '--------------------------------------------'
        print 'Giver ', self.name
        print '\t neighbours: ' % (self.neighbours)
        print '\t W0,W1,W2 : %.2f, %.2f, %.2f' %(self.W0, self.W1, self.W2)
        for nb in self.neighbours:
            print '\t with %5s: gave %.2f, recd %.2f' % (nb, self.given_val[nb], self.recd_val[nb])
        for x in self.world.commodities:
            print '\t %8s: %3d' % (x, self.count[x])
        print '\t utility = %.2f' %(self.get_utility())
        
    def do_one_gift(self, temperature = 1.0, verbose=False):
        """Consider giving one of a commodity to one neighbour. Evaluate
        drive to do this for all commodities and all neighbours,
        including a do-nothing option.

        Then DO IT (with the give() method)

        """
        if verbose: self.display()
        
        #%% first we decide what to do
        #print 'CHECK: %d nbrs and %d commods' % (len(self.neighbours), len(self.world.commodities))
        drive = np.zeros(shape=(len(self.neighbours), len(self.world.commodities)), dtype=float)
        base_utility = self.get_utility()   
        for i,c in enumerate(self.world.commodities):
            # eval the drive to give this c to this nb.
            # Thought experiment: "what if I gave c away to nb?"
            if self.count[c] > 0:
                self.count[c] -= 1 
                loss = self.get_utility() - base_utility
                self.count[c] += 1  # cos it's only a thought expt! 
                for j,nb in enumerate(self.neighbours):
                    drive[j,i] = np.exp(self.W0 +
                                                      self.W1 * loss + 
                                                      self.W2 * (self.recd_val[nb] - self.given_val[nb]))      
            else:
                for j,nb in enumerate(self.neighbours):
                    drive[j,i] = 0.0 # can't give away what don't have!

        if verbose: 
            print 'drive is ', drive
        
        total_drive = np.sum(np.ravel(drive))
        if verbose: 
            print '#### %s considers gifting ' % (self.name), self.world.commodities
            for j,nb in enumerate(self.neighbours):
                print '#### to %5s (drive) :' %(nb),
                print drive[j,:]
                print '#### to %5s (cumPr) :' %(nb),
                print prob #cumulativeProb.reshape(drive.shape)[j,:]

        # First decision: if do something what would it be?
        prob = drive / total_drive
        cumulativeProb = np.cumsum(prob)
        # choose an action with probability proportional to the normalised will.
        i = np.sum(rng.random() > cumulativeProb)
        nbr_index, commod_index = np.unravel_index(i, drive.shape)
        drive_of_chosen_act = drive[nbr_index, commod_index]

        # 2nd decision: do you want to do that, versus do nothing?
        drive_of_noaction = np.exp(1.)
        fraction = drive_of_noaction / (drive_of_noaction + drive_of_chosen_act)
        if rng.random() <  fraction:   # do nothing
            if verbose: print '#### best to PASS: %.2f better than %.2f' %(drive_of_noaction, drive_of_chosen_act)
        else:
            nbr = self.neighbours[nbr_index]
            commod = self.world.commodities[commod_index]
            if verbose: print '#### (%.2f) Choice: give %s to %s' % (r, commod, nbr)
                
            self.give(commod, nbr) # GIVE IT!

            if verbose: 
                self.display()
                nbr.display()


    def give(self, commodity, recipient):
        """ Give a specific commodity to a specific trader """
        if self.count[commodity] == 0:
            return
            
        base_utility = self.get_utility()   
        self.count[commodity]    -= 1
        lost = self.get_utility() - base_utility
        self.given_val[recipient] -= lost
    
        base_utility = recipient.get_utility()   
        recipient.count[commodity] += 1
        gained = recipient.get_utility() - base_utility
        recipient.recd_val[self] += gained

# ----------- end of Giver class definition ------------------------------------------

# ----------- start of World class definition ------------------------------------------

class World:
    def __init__(self, commodities=[]):
        self.commodities = commodities
        self.givers = []
        
    def add_node(self, g1, edges=[]):
        """ add a new node to the world, and (optionally) connect that node up """
        self.givers.append(g1)
        for g2 in edges:
            g1.add_neighbour(g2)
            g2.add_neighbour(g1)
        
    def do_one_season(self, verbose=False):
        """Zillions of trades happen. Agents accumulate rewards. 
    
        Should we have a fixed number of trades? Should rewards be grand sum,
        or on-the-fly exponentially weighted moving average?
        """
        # First, reset the initial amounts of stuff - e.g. reflecting stochasticity in the environment.
        for tr in self.givers:
            tr.utility_trail = [] # reset trail to blank - it's a new season.
            tr.count = {x: rng.randint(0,5)  for x in tr.world.commodities}
    
        # Second, do a bunch of rounds. In each round everyone gets to give if they want.
        for t in range(10):  # could probably be bigger!
            # do one round of all traders, in a random order
            for i in rng.permutation(len(self.givers)):
                tr = self.givers[i]
                tr.do_one_gift(1.0, verbose)
                tr.append_trail()
    
    def adapt_all_traders():
        """Everyone reconsiders their position, given their relative
        performance.  For example, perhaps everyone adopts the values and
        strategy (?) of someone else who is doing better. BUT HEY that
        won't do: if I'm in an apple-rich region do I suddenly want to
        value shoes? Sure, IF I DO then I might as well grab the
        shoe-guy's behavioural strategy too. But this seems to defy
        locality.
        """

# ----------- end of World class definition ------------------------------------------


#%% Other methods - the global stuff



if __name__ == '__main__':

    # Define the trader network
    world = World(['apples','shoes','wheat'])

    names = ['Bulbulia', 'Yoyo', 'Bianca', 'Mark', 'Serena', 'Kurt', 'Shaun', 'Shona', 'Grant', 'Bill', 'Shorty', 'Kim', 'Tyler']
    
    for name in names:
        world.add_node(Giver(world, name), edges=[])
    # link them up somehow...
    min_num_links = 2
    for tr1 in world.givers:
        n, attempts = 0, 0
        while (n < min_num_links) and (attempts < 10):
            attempts = attempts + 1
            tr2 = world.givers[rng.randint(len(world.givers))]
            isNewEdge = (tr1.add_neighbour(tr2) and tr2.add_neighbour(tr1))
            if isNewEdge:
                n = n + 1
    
    H = nx.frucht_graph()  # nice to use this one or other standards...
    # Test one trade:
    #tr = traders[rng.randint(len(traders))]
    #tr.do_one_gift()

    for tr in world.givers: 
        tr.display()
    print '^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^'
    world.do_one_season(verbose=False)
    for tr in world.givers: 
        tr.display()
    

    #%% Show the whole network as a picture    
    node_labels = []
    edge_ends_list = []
    edge_labels = []
    for tr1 in world.givers:
        node_labels.append(tr1.name)
        for tr2 in tr1.neighbours:
            # is it in the set already?
            if (tr2.name, tr1.name) not in edge_ends_list:
                edge_ends_list.append((tr1.name, tr2.name))
                edge_labels.append(tr1.name + ' and ' + tr2.name)
                
    G, node_posns = generate_drawable_graph(node_labels, edge_ends_list)
    edge_thck = 6
    #edge_thck = []
    #for (name1,name2) in edge_ends_list:
    #    edge_thck.append(traders[name1].num_trades[traders[name2]])
    draw_graph(G, node_posns, names, edge_ends_list, edge_labels, edge_thck)
