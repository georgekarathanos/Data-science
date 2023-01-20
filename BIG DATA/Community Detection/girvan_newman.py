import pandas as pd                ###FILE "fb-pages-food.edges" should be in the same directory with this code file.
import numpy as np
import random
import networkx as nx
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from networkx.algorithms import community as comm
from networkx.algorithms import centrality as cntr
import os
import itertools #new

############################### FG COLOR DEFINITIONS ###############################
class bcolors:
    HEADER      = '\033[95m'
    OKBLUE      = '\033[94m'
    OKCYAN      = '\033[96m'
    OKGREEN     = '\033[92m'
    WARNING     = '\033[93m'
    FAIL        = '\033[91m'
    ENDC        = '\033[0m'    # RECOVERS DEFAULT TEXT COLOR
    BOLD        = '\033[1m'
    UNDERLINE   = '\033[4m'

    def disable(self):
        self.HEADER     = ''
        self.OKBLUE     = ''
        self.OKGREEN    = ''
        self.WARNING    = ''
        self.FAIL       = ''
        self.ENDC       = ''

########################################################################################
############################## MY ROUTINES LIBRARY STARTS ##############################
########################################################################################

# SIMPLE ROUTINE TO CLEAR SCREEN BEFORE SHOWING A MENU...
def my_clear_screen():

    os.system('cls' if os.name == 'nt' else 'clear')

# CREATE A LIST OF RANDOMLY CHOSEN COLORS...
def my_random_color_list_generator(REQUIRED_NUM_COLORS):

    my_color_list = [   'red',
                        'green',
                        'cyan',
                        'brown',
                        'olive',
                        'orange',
                        'darkblue',
                        'purple',
                        'yellow',
                        'hotpink',
                        'teal',
                        'gold']

    my_used_colors_dict = { c:0 for c in my_color_list }     # DICTIONARY OF FLAGS FOR COLOR USAGE. Initially no color is used...
    constructed_color_list = []

    if REQUIRED_NUM_COLORS <= len(my_color_list):
        for i in range(REQUIRED_NUM_COLORS):
            constructed_color_list.append(my_color_list[i])
        
    else: # REQUIRED_NUM_COLORS > len(my_color_list)   
        constructed_color_list = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(REQUIRED_NUM_COLORS)]
 
    return(constructed_color_list)


# VISUALISE A GRAPH WITH COLORED NODES AND LINKS
def my_graph_plot_routine(G,fb_nodes_colors,fb_links_colors,fb_links_styles,graph_layout,node_positions):
    plt.figure(figsize=(10,8))
    
    if len(node_positions) == 0:
        if graph_layout == 'circular':
            node_positions = nx.circular_layout(G)
        elif graph_layout == 'random':
            node_positions = nx.random_layout(G, seed=50)
        elif graph_layout == 'planar':
            node_positions = nx.planar_layout(G)
        elif graph_layout == 'shell':
            node_positions = nx.shell_layout(G)
        else:   #DEFAULT VALUE == spring
            node_positions = nx.spring_layout(G)

    nx.draw(G, 
        with_labels=True,           # indicator variable for showing the nodes' ID-labels
        style=fb_links_styles,      # edge-list of link styles, or a single default style for all edges
        edge_color=fb_links_colors, # edge-list of link colors, or a single default color for all edges
        pos = node_positions,       # node-indexed dictionary, with position-values of the nodes in the plane
        node_color=fb_nodes_colors, # either a node-list of colors, or a single default color for all nodes
        node_size = 100,            # node-circle radius
        alpha = 0.9,                # fill-transparency 
        width = 0.5                 # edge-width
        )
    plt.show()

    return(node_positions)


########################################################################################
# MENU 1 STARTS: creation of input graph ### 
########################################################################################
def my_menu_graph_construction(G,node_names_list,node_positions):

    my_clear_screen()

    breakWhileLoop  = False

    while not breakWhileLoop:
        print(bcolors.OKGREEN 
        + '''
========================================
(1.1) Create graph from fb-food data set (fb-pages-food.nodes and fb-pages-food.nodes)\t[format: L,<NUM_LINKS>]
(1.2) Create RANDOM Erdos-Renyi graph G(n,p).\t\t\t\t\t\t[format: R,<number of nodes>,<edge probability>]
(1.3) Print graph\t\t\t\t\t\t\t\t\t[format: P,<GRAPH LAYOUT in {spring,random,circular,shell }>]    
(1.4) Continue with detection of communities.\t\t\t\t\t\t[format: N]
(1.5) EXIT\t\t\t\t\t\t\t\t\t\t[format: E]
----------------------------------------
        ''' + bcolors.ENDC)
        
        my_option_list = str(input('\tProvide your (case-sensitive) option: ')).split(',')
        
        if my_option_list[0] == 'L':
            MAX_NUM_LINKS = 2102    # this is the maximum number of links in the fb-food-graph data set...

            if len(my_option_list) > 2:
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                print(bcolors.WARNING + "\tERROR MESSAGE: Too many parameters. Try again..." + bcolors.ENDC)
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)            

            else:
                if len(my_option_list) == 1:
                    NUM_LINKS = MAX_NUM_LINKS
                else: #...len(my_option_list) == 2...
                    NUM_LINKS = int(my_option_list[1])

                if NUM_LINKS > MAX_NUM_LINKS or NUM_LINKS < 1:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(bcolors.WARNING + "\tERROR Invalid number of links to read from data set. It should be in {1,2,...,2102}. Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)            
                else:
                    # LOAD GRAPH FROM DATA SET...
                    G,node_names_list= read_graph_from_csv(NUM_LINKS)
                    print(  "\tConstructing the FB-FOOD graph with n =",G.number_of_nodes(),
                            "vertices and m =",G.number_of_edges(),"edges (after removal of loops).")

        elif my_option_list[0] == 'R':

            if len(my_option_list) > 3:
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                print(bcolors.WARNING + "\tERROR MESSAGE: Wrong number of parameters. Try again..." + bcolors.ENDC)
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
            
            else: # ...len(my_option_list) <= 3...
                if len(my_option_list) == 1:
                    NUM_NODES = 100                     # DEFAULT NUMBER OF NODES FOR THE RANDOM GRAPH...
                    ER_EDGE_PROBABILITY = 2 / NUM_NODES # DEFAULT VALUE FOR ER_EDGE_PROBABILITY...

                elif len(my_option_list) == 2:
                    NUM_NODES = int(my_option_list[1])
                    ER_EDGE_PROBABILITY = 2 / max(1,NUM_NODES) # AVOID DIVISION WITH ZERO...

                else: # ...NUM_NODES == 3...
                    NUM_NODES = int(my_option_list[1])
                    ER_EDGE_PROBABILITY = float(my_option_list[2])

                if ER_EDGE_PROBABILITY < 0 or ER_EDGE_PROBABILITY > 1 or NUM_NODES < 2:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(bcolors.WARNING + "\tERROR MESSAGE: Invalid probability mass or number of nodes of G(n,p). Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                else:
                    G = nx.erdos_renyi_graph(NUM_NODES, ER_EDGE_PROBABILITY)
                    print(bcolors.ENDC +    "\tConstructing random Erdos-Renyi graph with n =",G.number_of_nodes(),
                                            "vertices and edge probability p =",ER_EDGE_PROBABILITY,
                                            "which resulted in m =",G.number_of_edges(),"edges.")

                    node_names_list = [ x for x in range(NUM_NODES) ]

        elif my_option_list[0] == 'P':                  # PLOT G...
            print("Printing graph G with",G.number_of_nodes(),"vertices,",G.number_of_edges(),"edges and",nx.number_connected_components(G),"connected components." )
                
            if len(my_option_list) > 3:
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                print(bcolors.WARNING + "\tERROR MESSAGE: Wrong number of parameters. Try again..." + bcolors.ENDC)
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                
            else:
                if len(my_option_list) <= 1:
                    graph_layout = 'spring'     # ...DEFAULT graph_layout value...
                    reset_node_positions = 'Y'  # ...DEFAULT decision: erase node_positions...

                elif len(my_option_list) == 2: 
                    graph_layout = str(my_option_list[1])
                    reset_node_positions = 'Y'  # ...DEFAULT decision: erase node_positions...

                else: # ...len(my_option_list) == 3...
                    graph_layout = str(my_option_list[1])
                    reset_node_positions = str(my_option_list[2])

                if graph_layout not in ['spring','random','circular','shell']:
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                        print(bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible choice for graph layout. Try again..." + bcolors.ENDC)
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                elif reset_node_positions not in ['Y','y','N','n']:
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                        print(bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible decision for resetting node positions. Try again..." + bcolors.ENDC)
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                else:
                    if reset_node_positions in ['y','Y']:
                        node_positions = []         # ...ERASE previous node positions...

                    node_positions = my_graph_plot_routine(G,'grey','blue','solid',graph_layout,node_positions)

        elif my_option_list[0] == 'N':
            NUM_NODES = G.number_of_nodes()
            if NUM_NODES == 0:
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                print(bcolors.WARNING + "\tERROR MESSAGE: You have not yet constructed a graph to work with. Try again..." + bcolors.ENDC)
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
            else:
                my_clear_screen()
                breakWhileLoop = True
            
        elif my_option_list[0] == 'E':
            quit()

        else:
            print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
            print(bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible input was provided. Try again..." + bcolors.ENDC)
            print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

    return(G,node_names_list,node_positions)

########################################################################################
# MENU 2: detect communities in the constructed graph 
########################################################################################
def my_menu_community_detection(G,node_names_list,node_positions,hierarchy_of_community_tuples):

    breakWhileLoop = False

    community_tuples=[]
    while not breakWhileLoop:
            print(bcolors.OKGREEN 
                + '''
========================================
(2.1) Add random edges from each node\t\t\t[format: RE,<NUM_RANDOM_EDGES_PER_NODE>,<EDGE_ADDITION_PROBABILITY in [0,1]>]
(2.2) Add hamilton cycle (if graph is not connected)\t[format: H]
(2.3) Print graph\t\t\t\t\t[format: P,<GRAPH LAYOUT in { spring, random, circular, shell }>,<ERASE NODE POSITIONS in {Y,N}>]
(2.4) Compute communities with GIRVAN-NEWMAN\t\t[format: C,<ALG CHOICE in { O(wn),N(etworkx) }>,<GRAPH LAYOUT in {spring,random,circular,shell }>,<percent of nodes [0,100]>]
(2.5) Compute a binary hierarchy of communities\t\t[format: D,<NUM_DIVISIONS>,<GRAPH LAYOUT in {spring,random,circular,shell }>,<percent of nodes [0,100]>,{'1' for netowrkx implementation OR'2' for one-shot girvan_newman }]
(2.6) Compute modularity-values for all community partitions\t[format: M]
(2.7) Visualize the communities of the graph\t\t[format: V,<GRAPH LAYOUT in {spring,random,circular,shell}>]
(2.8) EXIT\t\t\t\t\t\t[format: E]
----------------------------------------
            ''' + bcolors.ENDC)

            my_option_list = str(input('\tProvide your (case-sensitive) option: ')).split(',')

            if my_option_list[0] == 'RE':                    # 2.1: ADD RANDOM EDGES TO NODES...

                if len(my_option_list) > 3:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(bcolors.WARNING + "\tERROR MESSAGE: Wrong number of parameters. [format: D,<NUM_RANDOM_EDGES>,<EDGE_ADDITION_PROBABILITY>]. Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                else:
                    if len(my_option_list) == 1:
                        NUM_RANDOM_EDGES = 1                # DEFAULT NUMBER OF RANDOM EDGES TO ADD (per node)
                        EDGE_ADDITION_PROBABILITY = 0.25    # DEFAULT PROBABILITY FOR ADDING EACH RANDOM EDGE (independently of other edges) FROM EAC NODE (independently from other nodes)...

                    elif len(my_option_list) == 2:
                        NUM_RANDOM_EDGES = int(my_option_list[1])
                        EDGE_ADDITION_PROBABILITY = 0.25    # DEFAULT PROBABILITY FOR ADDING EACH RANDOM EDGE (independently of other edges) FROM EAC NODE (independently from other nodes)...

                    else:
                        NUM_RANDOM_EDGES = int(my_option_list[1])
                        EDGE_ADDITION_PROBABILITY = float(my_option_list[2])
            
                    # CHECK APPROPIATENESS OF INPUT AND RUN THE ROUTINE...
                    if NUM_RANDOM_EDGES-1 not in range(5):
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                        print(bcolors.WARNING + "\tERROR MESSAGE: Too many random edges requested. Should be from {1,2,...,5}. Try again..." + bcolors.ENDC) 
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                    elif EDGE_ADDITION_PROBABILITY < 0 or EDGE_ADDITION_PROBABILITY > 1:
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                        print(bcolors.WARNING + "\tERROR MESSAGE: Not appropriate value was given for EDGE_ADDITION PROBABILITY. Try again..." + bcolors.ENDC)
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                    else:
                        add_random_edges_to_graph(G,node_names_list,NUM_RANDOM_EDGES,EDGE_ADDITION_PROBABILITY)

            elif my_option_list[0] == 'H':                  #2.2: ADD HAMILTON CYCLE...

                    add_hamilton_cycle_to_graph(G,node_names_list)

            elif my_option_list[0] == 'P':                  # 2.3: PLOT G...
                print("Printing graph G with",G.number_of_nodes(),"vertices,",G.number_of_edges(),"edges and",nx.number_connected_components(G),"connected components." )
                
                if len(my_option_list) > 2:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(bcolors.WARNING + "\tERROR MESSAGE: Wrong number of parameters. Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                
                else:
                    if len(my_option_list) <= 1:
                        graph_layout = 'spring'     # ...DEFAULT graph_layout value...

                    else: # ...len(my_option_list) == 2... 
                        graph_layout = str(my_option_list[1])

                    if graph_layout not in ['spring','random','circular','shell']:
                            print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                            print(bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible choice for graph layout. Try again..." + bcolors.ENDC)
                            print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                    else:
                        if len(my_option_list) == 2:
                            node_positions = []         # ...ERASE previous node positions...

                        node_positions = my_graph_plot_routine(G,'grey','blue','solid',graph_layout,node_positions)

            elif my_option_list[0] == 'C':      # 2.4: COMPUTE ONE-SHOT GN-COMMUNITIES
                NUM_OPTIONS = len(my_option_list)

                if NUM_OPTIONS > 4:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(bcolors.WARNING + "\tERROR MESSAGE: Wrong number of parameters. Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                else:
                    if NUM_OPTIONS == 1:
                        alg_choice  = 'N'            # DEFAULT COMM-DETECTION ALGORITHM == NX_GN
                        graph_layout = 'spring'     # DEFAULT graph layout == spring

                    elif NUM_OPTIONS == 2:
                        alg_choice  = str(my_option_list[1])
                        perc= 50 #deafult value for nodes percentage
                        graph_layout = 'spring'     # DEFAULT graph layout == spring
                
                    elif NUM_OPTIONS==3: # ...NUM_OPTIONS == 3...
                        alg_choice      = str(my_option_list[1])
                        graph_layout    = str(my_option_list[2])
                        perc= 50 #deafult value for nodes percentage

                    else:
                        alg_choice      = str(my_option_list[1])
                        graph_layout    = str(my_option_list[2])
                        perc= int(my_option_list[3])
                    # CHECKING CORRECTNESS OF GIVWEN PARAMETERS...
                    if alg_choice == 'N' and graph_layout in ['spring','circular','random','shell']:
                        community_tuples=use_nx_girvan_newman_for_communities(G)

                    elif alg_choice == 'O'and graph_layout in ['spring','circular','random','shell']:
                        community_tuples=one_shot_girvan_newman_for_communities(G,perc)

                    else:
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                        print(bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible parameters for executing the GN-algorithm. Try again..." + bcolors.ENDC)
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

            elif my_option_list[0] == 'D':          # 2.5: COMUTE A BINARY HIERARCHY OF COMMUNITY PARRTITIONS
                NUM_OPTIONS = len(my_option_list)
                NUM_NODES = G.number_of_nodes()
                NUM_COMPONENTS = nx.number_connected_components(G)
                #MAX_NUM_DIVISIONS = min( 8*NUM_COMPONENTS , np.floor(NUM_NODES/4) )

                if NUM_OPTIONS > 5:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(bcolors.WARNING + "\tERROR MESSAGE: Wrong number of parameters. Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                else:
                    if NUM_OPTIONS == 1:
                        number_of_divisions = 2*NUM_COMPONENTS      # DEFAULT number of communities to look for 
                        graph_layout = 'spring'                     # DEFAULT graph layout == spring
                        perc= 50                                    #default value for percentage of nodes
                        algo= 1                                     #default value algorithm choice
                        
                    elif NUM_OPTIONS == 2:
                        number_of_divisions = int(my_option_list[1])
                        graph_layout = 'spring'                     # DEFAULT graph layout == spring
                        perc= 50                                    #default value for percentage of nodes
                        algo= 1                                     #default value algorithm choice

                    
                    elif NUM_OPTIONS == 3:
                        number_of_divisions = int(my_option_list[1])
                        graph_layout = str(my_option_list[2])
                        perc= 50                                    #default value for percentage of nodes
                        algo= 1                                     #default value algorithm choice

                    elif NUM_OPTIONS == 4:
                        number_of_divisions = int(my_option_list[1])
                        graph_layout = str(my_option_list[2])
                        perc= int(my_option_list[3])
                        algo= 1                                     #default value algorithm choice

                    else:
                        number_of_divisions = int(my_option_list[1])
                        graph_layout = str(my_option_list[2])
                        perc= int(my_option_list[3])
                        algo= int(my_option_list[4])

                    # CHECKING SYNTAX OF GIVEN PARAMETERS...
                    if number_of_divisions < NUM_COMPONENTS:
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                        print(bcolors.WARNING + "\tERROR MESSAGE: The graph has already",NUM_COMPONENTS,"connected components. Try again..." + bcolors.ENDC)
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                    elif graph_layout not in ['spring','random','circular','shell']:
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                        print(bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible choice of a graph layout. Try again..." + bcolors.ENDC)
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                    elif perc<0 or perc>100:
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                        print(bcolors.WARNING + "\tProbability should be a number between 0 and 100. Try again..." + bcolors.ENDC)
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                    elif algo!=1 and algo!=2:
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                        print(bcolors.WARNING + "\tChoose 1 for one-shot or 2 for netowrkx implementation. Try again..." + bcolors.ENDC)
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    else:
                        hierarchy_of_community_tuples,community_tuples,graph_layout,node_positions= divisive_community_detection(G,number_of_divisions,graph_layout,[],perc,algo)

            elif my_option_list[0] == 'M':      # 2.6: DETERMINE PARTITION OF MIN-MODULARITY, FOR A GIVEN BINARY HIERARCHY OF COMMUNITY PARTITIONS
                community_tuples= determine_opt_community_structure(G,hierarchy_of_community_tuples,graph_layout,node_positions)


            elif my_option_list[0] == 'V':      # 2.7: VISUALIZE COMMUNITIES WITHIN GRAPH

                NUM_OPTIONS = len(my_option_list)

                if(len(community_tuples)==0):
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(bcolors.WARNING + "\tYou have not created communities yet. Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                elif NUM_OPTIONS > 2:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(bcolors.WARNING + "\tERROR MESSAGE: Wrong number of parameters. Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                
                else:

                    if NUM_OPTIONS == 1:
                        graph_layout = 'spring'                     # DEFAULT graph layout == spring
                    
                    else: # ...NUM_OPTIONS == 2...
                        graph_layout = str(my_option_list[1])

                    if graph_layout not in ['spring','random','circular','shell']:
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                        print(bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible choice of a graph layout. Try again..." + bcolors.ENDC)
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                    else:
                        visualize_communities(G,community_tuples,graph_layout,node_positions)

            elif my_option_list[0] == 'E':
                #EXIT the program execution...
                quit()

            else:
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                print(bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible input was provided. Try again..." + bcolors.ENDC)
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
    ### MENU 2 ENDS: detect communities in the constructed graph ### 

########################################################################################
############################### MY ROUTINES LIBRARY ENDS ############################### 
########################################################################################

########################################################################################
########################## ROUTINES LIBRARY STARTS ##########################
# FILL IN THE REQUIRED ROUTINES FROM THAT POINT ON...
########################################################################################

########################################################################################
def read_graph_from_csv(NUM_LINKS):

    fb_links= pd.read_csv("fb-pages-food.edges") #read edges
    lst1= fb_links["node_1"][0:NUM_LINKS] #keep only the first NUM_LINKS edges
    lst2= fb_links["node_2"][0:NUM_LINKS]
    final_1=[]
    final_2=[]
    node_names_list=[]
    fb_links_df= pd.DataFrame(list(zip(lst1, lst2)), columns =["node_1", "node_2"]) #create a new dataFrame with only the first NUM_LINKS edges
    for i in range(fb_links_df.shape[0]):
        if(fb_links_df["node_1"][i]!=fb_links_df["node_2"][i]): #add only the edges that are not loops
            final_1.append(fb_links_df["node_1"][i])
            final_2.append(fb_links_df["node_2"][i])
        if(not fb_links_df["node_1"][i] in node_names_list): #add the new nodes to the name list
            node_names_list.append(fb_links_df["node_1"][i])
        if(not fb_links_df["node_2"][i] in node_names_list): #add the new nodes to the name list
            node_names_list.append(fb_links_df["node_2"][i])
    fb_links_loopless_df= pd.DataFrame(list(zip(final_1, final_2)), columns =["node_1", "node_2"]) #final loopless links
    G = nx.from_pandas_edgelist(fb_links_loopless_df, "node_1", "node_2", create_using=nx.Graph()) #create graph
    return (G,node_names_list)
######################################################################################################################
def one_shot_girvan_newman_for_communities(G,percentage):

    start_time = time.time()

    community_tuples = [] #every element is a list with a connected_component
    components= sorted(nx.connected_components(G), key = len, reverse=True) #create components
    for cc in components: #keep components as lists and sorted by their size.
        community_tuples.append(list(cc))

    numOfNodes= len(community_tuples[0])*(percentage/100) #number of nodes we will take under consideration
    nodes=[] #nodes as roots to the bfs trees.
    j=0
    while(j<numOfNodes):
        nodes.append(community_tuples[0][j])
        j=j+1
    maxComp= G.subgraph(community_tuples[0]).copy()
    betweenness= nx.edge_betweenness_centrality_subset(maxComp,sources=nodes,targets=community_tuples[0],weight=None)
    sorted_betweenness= {k: v for k, v in sorted(betweenness.items(), key=lambda item: item[1], reverse= True)}
    list_form= list(sorted_betweenness.keys())
    i=0
    while(nx.is_connected(maxComp)):
        edge_to_remove= list_form[i]
        maxComp.remove_edge(edge_to_remove[0],edge_to_remove[1])
        G.remove_edge(edge_to_remove[0],edge_to_remove[1])
        i=i+1
    new_components= sorted(nx.connected_components(maxComp), key = len, reverse=True)
    community_tuples.pop(0)
    for lc in new_components:
        community_tuples.append(list(lc))
   
    end_time = time.time()

    print(bcolors.ENDC  + "\t===================================================")
    print(bcolors.ENDC  + "\tYOUR OWN computation of ONE-SHOT Girvan-Newman clustering for a graph with",G.number_of_nodes(),"nodes and",G.number_of_edges(),"links. "
                        + "Computation time =", end_time - start_time,"\n")
    return community_tuples
######################################################################################################################
# ...(b) USE NETWORKX IMPLEMENTATION OF ONE-SHOT GN-ALGORITHM...
######################################################################################################################
def use_nx_girvan_newman_for_communities(G):

    start_time = time.time()
    #At the beggining we have K components => K communities
    communities = comm.girvan_newman(G) #returns K+1 communities (breaks the biggest component to 2)

    community_tuples = []
    for com in next(communities): #save communities as lists of nodes
        community_tuples.append(list(com))

    end_time = time.time()
    print(bcolors.ENDC  + "\t===================================================")
    print(bcolors.ENDC  + "\tBUILT-IN computation of ONE-SHOT Girvan-Newman clustering for a graph with",G.number_of_nodes(),"nodes and",G.number_of_edges(),"links. "
                        + "Computation time =", end_time - start_time,"\n")
    return community_tuples
######################################################################################################################
def divisive_community_detection(G,number_of_divisions,graph_layout,node_positions,percentage,method):

    start_time = time.time()

    community_tuples = [] #every element is a list with a connected_component
    components= sorted(nx.connected_components(G), key = len, reverse=True)
    for cc in components:
        community_tuples.append(list(cc))
    numDiv= number_of_divisions
    while(len(community_tuples)>numDiv): #check if the divisions that were asked to happen where less than the number of components
        print("Wrong number of divisions... It should be at least ",len(community_tuples))
        numDiv= int(input("Give a new number: "))
    hierarchy_of_community_tuples=[] #list that holds list of communities
    hierarchy_of_community_tuples.append(tuple(community_tuples))
    if(method==1):
        while(len(community_tuples)<numDiv):
            sortListOfLists(community_tuples)
            maxComp= G.subgraph(community_tuples[0]).copy() #take the graph of the largest community
            new_communities=use_nx_girvan_newman_for_communities(maxComp)
            community_tuples.append(new_communities[0]) #add the new communities to the tuples
            community_tuples.append(new_communities[1])
            currentTriplet=(community_tuples[0],new_communities[0],new_communities[1])
            community_tuples.pop(0) #remove the old community from the tuples
            hierarchy_of_community_tuples.append(currentTriplet)
    else:
        while(len(community_tuples)<numDiv):
            sortListOfLists(community_tuples)
            maxComp= G.subgraph(community_tuples[0]).copy() #take the graph of the largest community
            new_communities=one_shot_girvan_newman_for_communities(maxComp,percentage)
            community_tuples.append(new_communities[0]) #add the new communities to the tuples
            community_tuples.append(new_communities[1])
            currentTriplet=(community_tuples[0],new_communities[0],new_communities[1])
            community_tuples.pop(0) #remove the old community from the tuples
            hierarchy_of_community_tuples.append(currentTriplet)

    end_time = time.time()
    print(bcolors.ENDC  + "\t===================================================")
    print(bcolors.ENDC  + "\tComputation of HIERARCHICAL BIPARTITION of G in communities, "
                        + "using the BUILT-IN girvan-newman algorithm, for a graph with",G.number_of_nodes(),"nodes and",G.number_of_edges(),"links. "
                        + "Computation time =", end_time - start_time,"\n")
    return (hierarchy_of_community_tuples,community_tuples,graph_layout,node_positions)
######################################################################################################################
def determine_opt_community_structure(G,hierarchy_of_community_tuples,graph_layout,node_positions):
    community_tuples=() #here we keep the picture of the graph after every division
    final_tuples_to_print=() #here we keep the communties with the max modularity value
    modularities=[] ##holds modularity for every division
    number_of_communities=[] ##holds the number of communities for every division
    modularity= -100
    i=0
    for i in range(len(hierarchy_of_community_tuples)):
        if(i==0):
            community_tuples=hierarchy_of_community_tuples[0] #start with the K communities
        else: #calculate K+1 communities
            lst= list(community_tuples)
            lst.remove(hierarchy_of_community_tuples[i][0]) #remove the bigger community
            lst.append(hierarchy_of_community_tuples[i][1]) #add the new community (1)
            lst.append(hierarchy_of_community_tuples[i][2]) #add the new community (2)
            community_tuples=tuple(lst)
        new_modularity= comm.modularity(G, community_tuples)
        modularities.append(new_modularity)
        if(modularity<new_modularity): #if we found the new max modularity refresh the value of the variant "modularity"
            modularity= new_modularity
            final_tuples_to_print=community_tuples
        number_of_communities.append(len(community_tuples))
        i=i+1
    fig = plt.figure(figsize = (10, 5))
    plt.bar(number_of_communities, modularities, color ='skyblue',width = 0.4)
    plt.grid(True, color = "grey", linewidth = "1.4", linestyle = "-")
    plt.xlabel("NUMBER OF COMMUNITIES IN PARTITION")
    plt.ylabel("MODAITY VALUE OF PARTITION")
    plt.title("Bar-Chart of Modality Values of Partiotions in Hierarchy")
    print("max modularity is: ",modularity)
    plt.show()
    visualize_communities(G,final_tuples_to_print,graph_layout,node_positions)
    return community_tuples

######################################################################################################################
def add_hamilton_cycle_to_graph(G,node_names_list):
    length_G=len(G)
    length_N=len(node_names_list)
    for i in range(length_N-1): #if it does not exist already. create edge from node i to node i+1
        if(not G.has_edge(node_names_list[i],node_names_list[i+1])):
            G.add_edge(node_names_list[i],node_names_list[i+1])
    if(not G.has_edge(node_names_list[-1],node_names_list[0])): #create edge from the last node to the first node so we have a circle
        G.add_edge(node_names_list[-1],node_names_list[0])

######################################################################################################################
# ADD RANDOM EDGES TO A GRAPH...
######################################################################################################################
def add_random_edges_to_graph(G,node_names_list,NUM_RANDOM_EDGES,EDGE_ADDITION_PROBABILITY):
    length_G=len(G)
    length_N=len(node_names_list)
    for i in range(length_G): #for every node in G
        if(i in G): #check if the node with new_id=i is not in graph (might be removed while reading the file)
            counter=0
            while(counter<NUM_RANDOM_EDGES): #try to create new NUM_RANDOM_EDGES from node i
                counter= counter+1
                randomNode= random.randint(0, length_N-1)
                if (not G.has_edge(i,node_names_list[randomNode])) and (i!=node_names_list[randomNode]): #check if edge already exists
                    prob= random.randint(0,9)
                    if(prob<EDGE_ADDITION_PROBABILITY*10): #create this edge with probability=EDGE_ADDITION_PROBABILITY
                        G.add_edge(i,node_names_list[randomNode])
    return G

######################################################################################################################
# VISUALISE COMMUNITIES WITHIN A GRAPH
######################################################################################################################
def visualize_communities(G,community_tuples,graph_layout,node_positions):    
    color_map= my_random_color_list_generator(len(community_tuples))
    final_colors=[]
    for node in G:
        for i in range(len(community_tuples)):
            if node in community_tuples[i]: #give to node the color of the community that it belongs
                final_colors.append(color_map[i])
                continue
    print("Showing ",len(color_map)," communities")
    my_graph_plot_routine(G,final_colors,"purple","solid",graph_layout,node_positions)

########################################################################################
########################### ROUTINES LIBRARY ENDS ###########################
########################################################################################
def sortListOfLists(array):
    for i in range(len(array)):
        for j in range(len(array)):
            if(len(array[i])>len(array[j])):
                tmp=array[j]
                array[j]=array[i]
                array[i]=tmp

########################################################################################
############################# MAIN MENUS OF USER CHOICES ############################### 
########################################################################################

############################### GLOBAL INITIALIZATIONS #################################
G = nx.Graph()                      # INITIALIZATION OF THE GRAPH TO BE CONSTRUCTED
node_names_list = []
node_positions = []                 # INITIAL POSITIONS OF NODES ON THE PLANE ARE UNDEFINED...
community_tuples = []               # INITIALIZATION OF LIST OF COMMUNITY TUPLES...
hierarchy_of_community_tuples = []  # INITIALIZATION OF HIERARCHY OF COMMUNITY TUPLES

G,node_names_list,node_positions = my_menu_graph_construction(G,node_names_list,node_positions)
my_menu_community_detection(G,node_names_list,node_positions,hierarchy_of_community_tuples)
