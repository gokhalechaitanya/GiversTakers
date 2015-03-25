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
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)



#%% ----------- start of Giver class definition ------------------------------------------
class Giver:

    def __init__(self, world, name, start_counts=None):
        self.world = world # the world this agent lives within.        
        self.name = name
        if start_counts is None:
            self.count = {x: rng.randint(0,5)  for x in self.world.commodities}            
        else:
            self.count = start_counts        
            
        self.W = rng.random(3)  - 0.5
        # W[0]: strength of tendency to PASS (avoid gifting at all)
        # W[1]: strength of tendency to base trades on the value being given away
        #            (a +ve W1 means agent prefers low-loss trades)
        # W[2]: strength of tendency to base trades on shared history of trades
        #            (a +ve W2 means agent prefers to trade when it is "in the red")

        self.neighbours = []
        # the keys for each of the following dicts will be the neighbours
        self.given_val = {}           
        self.recd_val = {}
        self.num_gifts_to = {}

        self.blank_memories() # ensure we start with a clean slate


    def blank_memories(self):
        self.given_val = {nb: 0.0  for nb in self.neighbours}
        self.recd_val = {nb: 0.0  for nb in self.neighbours}
        self.num_gifts_to = {nb: 0  for nb in self.neighbours}
        self.utility_trail = [] # just a record of utility over a season
        self.counts_trail = {c: []  for c in self.world.commodities}  # record for each commodity over a season


    def __str__(self):
        return self.name


    def get_utility(self):
        """
        return the utility to this agent of having the current commodity counts
        """
        util = 0.0
        wee = 1.00001 # perhaps having none is horrible but not fatal.
        for c in self.world.commodities:
            util += np.sum(np.power(0.75, np.arange(self.count[c])))
            """
            if self.count[c] >= 0:
                util += np.log2(wee + self.count[c])  # for example....
            else:
                # THIS SHOULD NEVER HAPPEN
                util += np.log2(wee)  # even worse to give away what you don't even have!
            """

        return util 


    def get_lookahead_utilities(self, incr=0):
        """ returns a dict with commodities as keys, and the post-gifting utility that 
            will follow if that commodity is incremented by incr.
            Incr will be -1 if self if giving, and +1 if it is being given to. """
        utils = {}
        mean = 0.0
        for c in self.world.commodities:
            # calc the util if this commod is altered by incr.
            self.count[c] += incr  # DO the thought experiment!
            utils[c] = self.get_utility()
            #print c, utils[c], self.count[c]
            mean += utils[c]
            self.count[c] -= incr  # UNDO the thought experiment!
        mean = mean / len(self.world.commodities)
        return utils, mean


    def get_nbr_histories(self): # returns a dict over neighbours
        nbr_history = {}
        mean = 0.0
        for nb in self.neighbours:
            nbr_history[nb] = self.recd_val[nb] - self.given_val[nb]
            mean += nbr_history[nb]
        mean = mean/len(self.neighbours)
        return nbr_history, mean        


    def append_trails(self):
        self.utility_trail.append(self.get_utility())
        for c in self.world.commodities:
            self.counts_trail[c].append(self.count[c])
                
    def add_neighbour(self, new_neighbour):
        if (new_neighbour in self.neighbours) or (new_neighbour.name == self.name):
            return False
        else:
            self.neighbours.append(new_neighbour)
            self.given_val[new_neighbour] = 0.0
            self.recd_val[new_neighbour]  = 0.0
            self.num_gifts_to[new_neighbour]  = 0
            return True

    def set_count(self, commodity, i): 
        self.count[commodity] = i

    def set_counts(self, dict_commods):
        """ 
        takes a dict of commodity counts and uses it to 
        set this agent's counts.
        """
        for c in dict_commods.keys():
            self.count[c] = dict_commods[c]

    def get_counts(self):
        return self.count

    def set_weights(self, w): 
        self.W = w

    def get_weights(self): 
        return self.W

    def display(self):
        print '--------------------------------------------'
        print 'Giver ', self.name,
        #print ' has neighbours ', [nb.name for nb in self.neighbours]

        print '\t  W: ' +  str(self.get_weights())
        for x in self.world.commodities:
            print '\t %10s: %3d' % (x, self.count[x])
        print '\t    utility: %.2f' %(self.get_utility())
        for nb in self.neighbours:
            print '\t with %5s: gave %.2f (%d gifts), recd %.2f (%d)' % (nb, self.given_val[nb], self.num_gifts_to[nb], self.recd_val[nb], nb.num_gifts_to[self])
        
        
    def do_one_gift(self, verbose = False):
        """Consider giving one of a commodity to one neighbour. Evaluate
        drive to do this for all commodities and all neighbours,
        including a do-nothing option.
        Then hand it over with the give() method.
        """
        gift = self.decide_gift(verbose = False)
        if gift != None:
            c, nb = gift
            self.give(c, nb) # GIVE IT!
            if verbose:
                print '\n>>>> %s gifts %s to %s, after which we have:' % (self.name,c,nb.name)
                self.display()
                nb.display()

    def decide_gift(self, verbose = False):
        """
        #    calc the one_unit_loss for each commodity. (nb: -inf if count is zero)
        #    ie. a function returning a vector of the utilities following each case.
        #    Then subtract the original to get the loss.
        #    Then make the mean of this zero? (just helps interpretation of W1 is all).
        #
        #    calc the "history" value for each neighbour (function returning a value)
        #
        #    form matrix of "desires" to gift X to i etc.
        #    choose one with prob.
        #    compare that one against "drive" to do nothing (=0).
        #    Make that second stochastic choice.
        #
        #    return the choice as tuple (commodity, neighbour), or None if there's no gift.
       """ 

        # Jump out immediately if gifting is impossible.        
        if (np.sum(list(self.count.values())) == 0) or (self.neighbours == None): 
            return None  # no gift even possible!
        
        # get the info together...
        #base_utility = self.get_utility()
        lookahead_util, mean_la_util = self.get_lookahead_utilities(-1) # returns a dict over commodities
        nbr_history, mean_history = self.get_nbr_histories() # returns a dict over neighbours

        # form a matrix of "desires" to give X to i, etc.
        temperature = 1.0
        drive = np.zeros(shape=(len(self.neighbours), len(self.world.commodities)), dtype=float)
        for row, nb in enumerate(self.neighbours):
            for col, c in enumerate(self.world.commodities):
                phi  = self.W[0] + self.W[1] * (lookahead_util[c] - mean_la_util) + self.W[2] * nbr_history[nb]
                drive[row, col] = np.exp(phi/temperature)
        if verbose: 
            print '#### %s considers gifting ' % (self.name), self.world.commodities
            for j,nb in enumerate(self.neighbours):
                print '#### to %5s (drive) :' %(nb),
                print drive[j,:]
                print '#### to %5s (cumPr) :' %(nb),

        # First decision: if do something what would it be?
        total_drive = np.sum(np.ravel(drive))
        if total_drive == 0.0:
            print 'oh shit!',
            print np.ravel(drive)
        prob = drive / total_drive
        cumulativeProb = np.cumsum(prob)
        # choose an action with probability proportional to the normalised will.
        i = np.sum(rng.random() > cumulativeProb)
        nbr_index, commod_index = np.unravel_index(i, drive.shape)
        drive_of_chosen_act = drive[nbr_index, commod_index]

        # 2nd decision: do you want to do that, versus do nothing?
        drive_of_noaction = 1.0 # ie. exp(0).
        fraction = drive_of_noaction / (drive_of_noaction + drive_of_chosen_act)
        if rng.random() <  fraction:   # do nothing
            if verbose: print '#### decided to PASS'
            return None # could have gifted, but decided not to.
        else:
            nbr = self.neighbours[nbr_index]
            commod = self.world.commodities[commod_index]
            if self.count[commod] <= 0:
                return None
            if verbose: print '#### (%.2f) Choice: give %s to %s' % (r, commod, nbr)
            return commod, nbr
            

    def give(self, commodity, recipient):
        """ Give a specific commodity to a specific trader """
        if self.count[commodity] == 0:
            return

        base_utility = self.get_utility()   
        self.count[commodity]    -= 1
        lost = self.get_utility() - base_utility
        self.given_val[recipient] = self.given_val[recipient] - lost
        self.num_gifts_to[recipient] += 1
    
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
        
    def do_one_season(self, num_gifts=1, verbose=False):
        """Gifts happen. Agents accumulate rewards. 
        """
        # First, reset the initial amounts of stuff - e.g. reflecting stochasticity in the environment.
        for tr in self.givers:
            tr.blank_memories() # reset trail to blank - it's a new season.
    
        # Second, do a bunch of rounds. In each round everyone gets to give if they want.
        for t in range(num_gifts):  # could probably be bigger!
            # do one round of all traders, in a random order
            for i in rng.permutation(len(self.givers)):
                tr = self.givers[i]
                tr.do_one_gift(verbose)
                tr.append_trails()
    
    def adapt_all_traders():
        """Everyone reconsiders their position, given their relative
        performance.  For example, perhaps everyone adopts the values and
        strategy (?) of someone else who is doing better. BUT HEY that
        won't do: if I'm in an apple-rich region do I suddenly want to
        value shoes? Sure, IF I DO then I might as well grab the
        shoe-guy's behavioural strategy too. But this seems to defy
        locality.
        """

    def show_network(self, filename=None):
        #%% Show the whole network as a picture    
        node_labels_list = []
        edge_ends_list = []
        edge_labels_list = []
        edge_thck = []

        for tr1 in self.givers:
            node_labels_list.append(tr1.name)
            for tr2 in tr1.neighbours:
                # is it in the set already?
                if (tr2.name, tr1.name) not in edge_ends_list:
                    edge_ends_list.append((tr1.name, tr2.name))
                    edge_labels_list.append(tr1.name + ' and ' + tr2.name)
                    flux_estimate = tr1.num_gifts_to[tr2] + tr2.num_gifts_to[tr1] 
                    edge_thck.append(flux_estimate)

        # rescale the node sizes to between 0 and 1000, say.
        # Use the final utility:
        node_sizes = np.array([int(100*g.get_utility()) for g in self.givers])
        # Or take an average over all time:
        #node_sizes = np.array([int(100*np.mean(g.utility_trail)) for g in self.givers])
        node_sizes = np.power(node_sizes, 2.0)
        #node_sizes = node_sizes - node_sizes.min()
        node_sizes_list = list((node_sizes * 950/node_sizes.max()) + 50)

        # rescale the edge thicknesses to between 0 and 10, say.
        tmp = np.array(edge_thck)
        if tmp.max() <= 0.0:
            edge_thck = 6 # abandon all hope and just use "sucks" for all.
        else:
            #tmp = tmp - tmp.min()
            edge_thck = list(tmp * 10/tmp.max()  + 0.3)

        G, node_posns = generate_drawable_graph(node_labels_list, edge_ends_list)
        draw_graph(G, node_posns, node_labels_list, node_sizes_list, edge_ends_list, edge_labels_list, edge_thck, filename=filename)


    def display_pair_sequence(self, playerA, playerB, outfile = 'apair.png'):
        # Display the hell out of it
        plt.clf()
        plt.subplot(3,1,1)
        plt.plot(playerA.utility_trail,'s-k')
        plt.plot(playerB.utility_trail,'o-k', markerfacecolor='white', alpha=.5)
        plt.title('utilities over time')
        biplayerA = np.max(playerA.utility_trail)
        biplayerB = np.max(playerB.utility_trail)
        biggest = max(biplayerA, biplayerB)
        #plt.gca().set_ylim(-0.5, biggest + .5)
        
        maxcount = max(np.max([playerA.counts_trail[x] for x in self.commodities]), np.max([playerB.counts_trail[x] for x in self.commodities]))

        plt.subplot(3,1,2) # player A
        for x in self.commodities:
            plt.plot(playerA.counts_trail[x],'-s', alpha=.5, label = x)
        plt.gca().set_ylim(-0.5,maxcount+0.5)
        plt.ylabel(playerA.name)
        plt.gca().set_xticks([])
        l = plt.legend()

        plt.subplot(3,1,3) # player B
        for x in self.commodities:
            plt.plot(playerB.counts_trail[x],'-o', alpha=.5, label = x)
        plt.gca().set_ylim(-0.5,maxcount+0.5)
        plt.ylabel(playerB.name)
        plt.gca().set_xticks([])
        plt.savefig(outfile,dpi=200)
        print 'wrote %s' % (outfile)


    def display_all_sequences(self, outfile = 'all_sequences.png'):
        # Display the hell out of it
        plt.clf()
        N = len(self.givers)

        # figure out the largest count so all axes can use it / be comparable.
        maxcount = 0
        for g in self.givers:
            maxcount = max(maxcount, np.max([g.counts_trail[x] for x in self.commodities]))
        minutility, maxutility = 10000.0, -100000.0
        for g in self.givers:
            maxutility = max(maxutility,  max(g.utility_trail))
            minutility = min(minutility,  min(g.utility_trail))
        print   'min and max : ', minutility, maxutility

        n = 1
        for g in self.givers:
            plt.subplot(N,1,n)
            n = n+1

            L = len(g.counts_trail[x])
            for i in range(L):
                alph = 1.0 * float(g.utility_trail[i] - minutility) / (maxutility - minutility)
                plt.plot(i*np.ones(maxcount/2), range(0,maxcount, 2),'o', color='pink', markeredgecolor='none', alpha=alph)
            for x in self.commodities:
                plt.plot(g.counts_trail[x],'-s', alpha=1.0, label = x)
            plt.gca().set_ylim(-0.5,maxcount+0.5)
            plt.ylabel(g.name)
            plt.gca().set_xticks([])

        l = plt.legend()
        plt.savefig(outfile,dpi=200)
        print 'wrote %s' % (outfile)
        return

# ----------- end of World class definition ------------------------------------------


#%% Other methods - the global stuff
def run_two_givers(playA, playB, num_gifts):
    # Blank all memory of previous transactions
    playA.blank_memories()
    playB.blank_memories()

    for t in range(num_gifts):
        playA.append_trails()
        playB.append_trails()

        if rng.random() > 0.5:
            playA.do_one_gift()
        else:
            playB.do_one_gift()

    return




##############################################    


if __name__ == '__main__':

    # Define the trader network
    world = World(['apples', 'shoes', 'wheat'])

    names = ['Shaun', 'Andy', 'Tava', 'Alexei', 'Stephen', 'Marcus', 'Sally', 'Dion']
    
    for name in names:
        tmpg = Giver(world, name, None)
        world.add_node(tmpg, edges=[])
    # link them up somehow...
    min_num_links = 2
    for tr1 in world.givers:
        print 'wiring up ', tr1.name
        n, attempts = 0, 0
        while (n < min_num_links) and (attempts < 10):
            attempts = attempts + 1
            tr2 = world.givers[rng.randint(len(world.givers))]
            isNewEdge = (tr1.add_neighbour(tr2) and tr2.add_neighbour(tr1))
            if isNewEdge:
                print '\t wired up ', tr1.name, ' to ', tr2.name
                n = n + 1
        print tr1.name, ' should be done - let us see: '
        tr1.display()
    print '^^^^^^^^^^^^^^^^   all wired up ^^^^^^^^^^^^^^^^^^^^^^'
    
    H = nx.frucht_graph()  # nice to use this one or other standards...
    # Test one trade:
    #tr = traders[rng.randint(len(traders))]
    #tr.do_one_gift()

    print 'doing one season...'
    world.do_one_season(num_gifts=10, verbose=False)
    for tr in world.givers: 
        tr.display()
    
    print 'showing the network...'
    world.show_network('givers_random_network_example.png')
