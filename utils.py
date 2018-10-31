"""
V3.0 
Provides some widely useful utilities -Based on AIMA code
We will incrementally build this through the course...
"""
import operator, math, random, copy, sys, os.path, bisect, re
from functools import reduce

#______________________________________________________________________________
# Simple Data Structures

infinity = 1.0e400

def Dict(**entries):
    """Create a dict out of the argument=value arguments.
    >>> Dict(a=1, b=2, c=3)
    {'a': 1, 'c': 3, 'b': 2}
    """
    return entries  #converts multiple name value pairs to a dictionary

class DefaultDict(dict):
    """Dictionary with a default value for unknown keys."""
    def __init__(self, default):
        self.default = default

    def __getitem__(self, key):
        if key in self: return self.get(key)
        return self.setdefault(key, copy.deepcopy(self.default))

    def __copy__(self):
        copy = DefaultDict(self.default)
        copy.update(self)
        return copy
        
def update(x, **entries):
    """Update a dict, or an object with slots, according to `entries` dict.
    >>> update({'a': 1}, a=10, b=20)
    {'a': 10, 'b': 20}
    """
    if isinstance(x, dict):
        x.update(entries)
    else:
        x.__dict__.update(entries) #update method updates dictionary with another's keys
    return x

#______________________________________________________________________________
# Misc Functions
def euclidean(a, b): 
    '''The Euclidean distance between two (x, y) points.'''
    return math.hypot((a[0] - b[0]), (a[1] - b[1]))

def print_table(table, header=None, key=None):
    """Print a list of lists as a table, so that columns line up nicely.
    header, if specified, will be printed as the first row."""
    if key:
        print (key)
    if header:
        table = [header] + table
    maxlen = lambda seq: max(map(len, seq))
    sizes = map(maxlen, zip(*[map(str, row) for row in table]))
    alignments =[]
    for row in table:
        for (size,x) in zip(sizes,row):
            alignments.append(size) 
    for row in table:
        i=0
        for s in row:
            print(repr(s).ljust(alignments[i]+3), end="")
            i+=1
        print()
    print()

def memoize(fn, slot=None):
    """Memoize fn: make it remember the computed value for any argument list.
    If slot is specified, store result in that slot of first argument.
    If slot is false, store results in a dictionary."""
    if slot:
        def memoized_fn(obj, *args):
            if hasattr(obj, slot):
                return getattr(obj, slot)
            else:
                val = fn(obj, *args)
                setattr(obj, slot, val)
                return val
    else:
        def memoized_fn(*args):
            if not memoized_fn.cache.has_key(args):
                memoized_fn.cache[args] = fn(*args)
            return memoized_fn.cache[args]
        memoized_fn.cache = {}
    return memoized_fn

#______________________________________________________________________________
# Functions on Sequences (mostly inspired by Common Lisp)
# NOTE: Sequence functions (count_if, find_if, every, some) take function
# argument first (like reduce, filter, and map).

def removeall(item, seq):
    """Return a copy of seq (or string) with all occurences of item removed.
    >>> removeall(3, [1, 2, 3, 3, 2, 1, 3])
    [1, 2, 2, 1]
    >>> removeall(4, [1, 2, 3])
    [1, 2, 3]
    """
    if isinstance(seq, str):
        return seq.replace(item, '')
    else:
        return [x for x in seq if x != item]

def unique(seq):
    """Remove duplicate elements from seq. Assumes hashable elements.
    >>> unique([1, 2, 3, 2, 1])
    [1, 2, 3]
    """
    return list(set(seq))

def product(numbers):
    """Return the product of the numbers.
    >>> product([1,2,3,4])
    24
    """
    return reduce(operator.mul, numbers, 1)

def count_if(predicate, seq):
    """Count the number of elements of seq for which the predicate is true.
    >>> count_if(callable, [42, None, max, min])
    2
    """
    f = lambda count, x: count + (not not predicate(x))
    return reduce(f, seq, 0)

def find_if(predicate, seq):
    """If there is an element of seq that satisfies predicate; return it.
    >>> find_if(callable, [3, min, max])
    <built-in function min>
    >>> find_if(callable, [1, 2, 3])
    """
    for x in seq:
        if predicate(x): return x
    return None

def every(predicate, seq):
    """True if every element of seq satisfies predicate.
    >>> every(callable, [min, max])
    1
    >>> every(callable, [min, 3])
    0
    """
    for x in seq:
        if not predicate(x): return False
    return True

def some(predicate, seq):
    """If some element x of seq satisfies predicate(x), return predicate(x).
    >>> some(callable, [min, 3])
    1
    >>> some(callable, [2, 3])
    0
    """
    for x in seq:
        px = predicate(x)
        if px: return px
    return False

def isin(elt, seq):
    """Like (elt in seq), but compares with is, not ==.
    >>> e = []; isin(e, [1, e, 3])
    True
    >>> isin(e, [1, [], 3])
    False
    """
    for x in seq:
        if elt is x: return True
    return False

#______________________________________________________________________________
# Functions on sequences of numbers
# NOTE: these take the sequence argument first, like min and max,
# and like standard math notation: \sigma (i = 1..n) fn(i)
# A lot of programing is finding the best value that satisfies some condition;
# so there are three versions of argmin/argmax, depending on what you want to
# do with ties: return the first one, return them all, or pick at random.

def argmin(seq, fn):
    """Return an element with lowest fn(seq[i]) score; tie goes to first one.
    >>> argmin(['one', 'to', 'three'], len)
    'to'
    """
    best = seq[0]; best_score = fn(best)
    for x in seq:
        x_score = fn(x)
        if x_score < best_score:
            best, best_score = x, x_score
    return best

def argmin_list(seq, fn):
    """Return a list of elements of seq[i] with the lowest fn(seq[i]) scores.
    >>> argmin_list(['one', 'to', 'three', 'or'], len)
    ['to', 'or']
    """
    best_score, best = fn(seq[0]), []
    for x in seq:
        x_score = fn(x)
        if x_score < best_score:
            best, best_score = [x], x_score
        elif x_score == best_score:
            best.append(x)
    return best

def argmin_random_tie(seq, fn):
    """Return an element with lowest fn(seq[i]) score; break ties at random.
    Thus, for all s,f: argmin_random_tie(s, f) in argmin_list(s, f)"""
    best_score = fn(seq[0]); n = 0
    for x in seq:
        x_score = fn(x)
        if x_score < best_score:
            best, best_score = x, x_score; n = 1
        elif x_score == best_score:
            n += 1
            if random.randrange(n) == 0:
                best = x
    return best

def argmax(seq, fn):
    """Return an element with highest fn(seq[i]) score; tie goes to first one.
    >>> argmax(['one', 'to', 'three'], len)
    'three'
    """
    return argmin(seq, lambda x: -fn(x))

def argmax_list(seq, fn):
    """Return a list of elements of seq[i] with the highest fn(seq[i]) scores.
    >>> argmax_list(['one', 'three', 'seven'], len)
    ['three', 'seven']
    """
    return argmin_list(seq, lambda x: -fn(x))

def argmax_random_tie(seq, fn):
    "Return an element with highest fn(seq[i]) score; break ties at random."
    return argmin_random_tie(seq, lambda x: -fn(x)) 

#______________________________________________________________________________
#Statistical functions
def probability(p):
    "Return true with probability p."
    return p > random.uniform(0.0, 1.0)

def weighted_sample_with_replacement(seq, weights, n):
    """Pick n samples from seq at random, with replacement, with the
    probability of each element in proportion to its corresponding
    weight."""
    sample = weighted_sampler(seq, weights)
    return [sample() for s in range(n)]

def weighted_sampler(seq, weights):
    "Return a random-sample function that picks from seq weighted by weights."
    totals = []
    for w in weights:
        totals.append(w + totals[-1] if totals else w)
    return lambda: seq[bisect.bisect(totals, random.uniform(0, totals[-1]))]
    
#______________________________________________________________________________
# Queues: Stack, FIFOQueue

class Queue:
    """Queue is an abstract class/interface. There are three types:
        Stack(): A Last In First Out Queue.
        FIFOQueue(): A First In First Out Queue.
    Each type supports the following methods and functions:
        q.append(item)  -- add an item to the queue
        q.extend(items) -- equivalent to: for item in items: q.append(item)
        q.pop()         -- return the top item from the queue
        len(q)          -- number of items in q (also q.__len())
        item in q       -- does q contain item?
    Note that isinstance(Stack(), Queue) is false, because we implement stacks
    as lists.  If Python ever gets interfaces, Queue will be an interface."""

    def __init__(self):
        abstract

    def extend(self, items):
        for item in items: self.append(item)

def Stack():
    """Return an empty list, suitable as a Last-In-First-Out Queue."""
    return []

class FIFOQueue(Queue):
    """A First-In-First-Out Queue."""
    def __init__(self):
        self.A = []; self.start = 0
    def append(self, item):
        self.A.append(item)
    def __len__(self):
        return len(self.A) - self.start
    def extend(self, items):
        self.A.extend(items)
    def pop(self):
        e = self.A[self.start]
        self.start += 1
        if self.start > 5 and self.start > len(self.A)/2:
            self.A = self.A[self.start:]
            self.start = 0
        return e
    def __contains__(self, item):
        return item in self.A[self.start:]

class PriorityQueue(Queue):
    """A queue in which the minimum (or maximum) element (as determined by f and
    order) is returned first. If order is min, the item with minimum f(x) is
    returned first; if order is max, then it is the item with maximum f(x).
    Also supports dict-like lookup."""
    def __init__(self, order=min, f=lambda x: x):
        update(self, A=[], order=order, f=f)

    def append(self, item):
        bisect.insort(self.A, (self.f(item), item))

    def __len__(self):
        return len(self.A)

    def pop(self):
        if self.order == min:
            return self.A.pop(0)[1]
        else:
            return self.A.pop()[1]

    def __contains__(self, item):
        return some(lambda x: x[1] == item, self.A)

    def __getitem__(self, key):
        for _, item in self.A:
            if item == key:
                return item

    def __delitem__(self, key):
        for i, (value, item) in enumerate(self.A):
            if item == key:
                self.A.pop(i)
                return




