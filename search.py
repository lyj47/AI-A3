"""
V3.0
Code is based on simplified AIMA book code repository, updated for python 3.
The way to use this code is to subclass Problem to create a class of problems,
then create problem instances and solve them with calls to the various search
functions."""

from utils import PriorityQueue, FIFOQueue, Stack, Dict
from utils import update,euclidean,infinity,print_table,memoize 
import sys

#______________________________________________________________________________

class Problem(object):
    """The abstract class for a formal problem.  You should subclass
    this and implement the methods actions and result, and possibly
    __init__, goal_test, and path_cost. Then you will create instances
    of your subclass and solve them with the various search functions."""

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal.  Your subclass's constructor can add
        other arguments."""
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        abstract #There is no keyword abstract in python. This is just a trick that AIMA uses

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        abstract

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal, as specified in the constructor. Override this
        method if checking against a single self.goal is not enough."""
        return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2.  If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

#______________________________________________________________________________


class Node:
    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state.  Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node. You will not need to
    subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        "Create a search tree Node, derived from a parent by an action."
        update(self, state=state, parent=parent, action=action,
               path_cost=path_cost, depth=0)
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node %s>" % (self.state,)

    def expand(self, problem):
        "List the nodes reachable in one step from this node."
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]
        
    def child_node(self, problem, action):
        next = problem.result(self.state, action)
        return Node(next, self, action,
                    problem.path_cost(self.path_cost, self.state, action, next))

    def solution(self):
        "Return the sequence of actions to go from the root to this node."
        return [node.action for node in self.path()[1:]]

    def path(self):
        "Return a list of nodes forming the path from the root to this node."
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    # We want a queue/list of nodes to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in many contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(str(self.state))

    #Added V2.0
    def __lt__(self, other):
        return self.f < other.f

#______________________________________________________________________________
# Uninformed Search algorithms

def graph_search(problem, frontier):
    """Search through the successors of a problem to find a goal.
    The argument frontier should be an empty queue.
    If two paths reach a state, only use the first one. """
    frontier.append(Node(problem.initial))
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        explored.add(node.state)
        frontier.extend(child for child in node.expand(problem)
                        if child.state not in explored)
    return None

def depth_first_search(problem):
    "Search the deepest nodes in the search tree first."
    return graph_search(problem, Stack())

def breadth_first_search(problem):
    "Search the shallowest nodes in the search tree first."
    return graph_search(problem, FIFOQueue())
   
def depth_limited_search(problem, limit=50):
    def recursive_dls(node, problem, limit):
        if problem.goal_test(node.state):
            return node
        elif node.depth == limit:
            return 'cutoff'
        else:
            cutoff_occurred = False
            for child in node.expand(problem):
                result = recursive_dls(child, problem, limit)
                if result == 'cutoff':
                    cutoff_occurred = True
                elif result is not None:
                    return result
            if cutoff_occurred:               
                return 'cutoff'
            else:
                return None

    # Body of depth_limited_search:
    return recursive_dls(Node(problem.initial), problem, limit)

def iterative_deepening_search(problem):
    for depth in range(sys.maxsize):
        result = depth_limited_search(problem, depth)
        if result != 'cutoff':
            return result
        else:
            print ("Cutoff occurred for depth ",depth)

#Added in V2.0
#______________________________________________________________________________
# Informed (Heuristic) Search
def best_first_graph_search(problem, f):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""
    f = memoize(f, 'f')
    node = Node(problem.initial)
    if problem.goal_test(node.state):
        return node
    frontier = PriorityQueue(min, f)
    frontier.append(node)
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                incumbent = frontier[child]
                if f(child) < f(incumbent):
                    del frontier[incumbent]
                    frontier.append(child)
    return None

def uniform_cost_search(problem):
    return best_first_graph_search(problem, lambda node: node.path_cost)
    
def greedy_best_first_graph_search(problem, h=None):
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, lambda n: h(n))
	
def astar_search(problem, h=None):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, lambda n: n.path_cost + h(n))

#Added in V2.0
#______________________________________________________________________________
# Graphs and Graph Problems

class Graph:
    """A graph connects nodes (vertices) by edges (links).  Each edge can also
    have a length associated with it.  The constructor call is something like:
        g = Graph({'A': {'B': 1, 'C': 2}})
    this makes a graph with 3 nodes, A, B, and C, with an edge of length 1 from
    A to B,  and an edge of length 2 from A to C.  You can also do:
        g = Graph({'A': {'B': 1, 'C': 2}}, directed=False)
    This makes an undirected graph, so inverse links are also added. The graph
    stays undirected; if you add more links with g.connect('B', 'C', 3), then
    inverse link is also added.  You can use g.nodes() to get a list of nodes,
    g.get('A') to get a dict of links out of A, and g.get('A', 'B') to get the
    length of the link from A to B.  'Lengths' can actually be any object at
    all, and nodes can be any hashable object."""

    def __init__(self, dict=None, directed=True):
        self.dict = dict or {}
        self.directed = directed
        if not directed: self.make_undirected()

    def make_undirected(self):
        "Make a digraph into an undirected graph by adding symmetric edges."
        dict1 = dict(self.dict) # have to make copy because of python 3.0 limitations
        for a in dict1.keys():
            for (b, distance) in dict1[a].items():
                self.connect1(b, a, distance)

    def connect(self, A, B, distance=1):
        """Add a link from A and B of given distance, and also add the inverse
        link if the graph is undirected."""
        self.connect1(A, B, distance)
        if not self.directed: self.connect1(B, A, distance)

    def connect1(self, A, B, distance):
        "Add a link from A to B of given distance, in one direction only."
        self.dict.setdefault(A,{})[B] = distance

    def get(self, a, b=None):
        """Return a link distance or a dict of {node: distance} entries.
        .get(a,b) returns the distance or None;
        .get(a) returns a dict of {node: distance} entries, possibly {}."""
        links = self.dict.setdefault(a, {})
        if b is None: return links
        else: return links.get(b)

    def nodes(self):
        "Return a list of nodes in the graph."
        return self.dict.keys()

def UndirectedGraph(dict=None):
    "Build a Graph where every edge (including future ones) goes both ways."
    return Graph(dict=dict, directed=False)

class GraphProblem(Problem):
    "The problem of searching a graph from one node to another."
    def __init__(self, initial, goal, graph):
        Problem.__init__(self, initial, goal)
        self.graph = graph

    def actions(self, A):
        "The actions at a graph node are just its neighbors."
        return self.graph.get(A).keys()

    def result(self, state, action):
        "The result of going to a neighbor is just that neighbor."
        return action

    def path_cost(self, cost_so_far, A, action, B):
        return cost_so_far + (self.graph.get(A,B) or infinity)

    def h(self, node):
        "h function is straight-line distance from a node's state to goal."
        "Override this to be something meaningful in your graph domain"
        locs = getattr(self.graph, 'locations', None)
        if locs:
            return euclidean(locs[node.state], locs[self.goal])
        else:
            return infinity

#______________________________________________________________________________
# Code to compare searchers on various problems.
class InstrumentedProblem(Problem):
    """Delegates to a problem, and keeps statistics."""

    def __init__(self, problem):
        #succs - nodes expanded 
        #states - nodes generated
        #goal_tests - number of times goal is tested
        self.problem = problem
        self.succs = self.goal_tests = self.states = self.final_cost = 0
        self.found = None

    def actions(self, state):
        self.succs += 1
        return self.problem.actions(state)

    def result(self, state, action):
        self.states += 1
        return self.problem.result(state, action)

    def goal_test(self, state):
        self.goal_tests += 1
        result = self.problem.goal_test(state)
        if result:
            self.found = state
        return result

    def path_cost(self, c, state1, action, state2):
        return self.problem.path_cost(c, state1, action, state2)

 
    def __getattr__(self, attr):
        return getattr(self.problem, attr)

    def __repr__(self):
        return '<%4d/%4d/%4d/%4d/%s>' % (self.succs, self.goal_tests,
                                     self.states, self.final_cost, str(self.found)[:4])


def name(obj):
    try:
        return obj.__name__
    except:
        return str(obj)


def compare_searchers(problems, header,
                      searchers=[
                                 breadth_first_search, depth_first_search,
                                 iterative_deepening_search,]):
    def do(searcher, problem):
        p = InstrumentedProblem(problem)
        g=searcher(p)
        p.final_cost=g.path_cost
        return p
    table = [[name(s)] + [do(s, p) for p in problems] for s in searchers]
    print_table(table, header,"Expanded/Goal Tests/Generated/Cost/Goal Found")
#______________________________________________________________________________
