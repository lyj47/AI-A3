
from search import Problem
from csp import backtracking_search, no_inference, forward_checking, mac
from csp import first_unassigned_variable,mrv,num_legal_values,unordered_domain_values,lcv
from csp import CSP,usa,france,australia,NQueen,UniversalDict

#______________________________________________________________________________
#An implementation of Homework 7
vars = ['A','B','C','D','E']
neighbors = {'A': ['B','D','E'], 'B': ['A','E'], 'C': ['D','E'],
             'D': ['A','C','E'], 'E': ['A','B','C','D']} 
#domains = {'A': [1,2,3,4],'B': [1,2,3,4],'C': [1,2,3,4],'D': [1,2,3,4],'E': [1,2,3,4],} 
#Since they all have the same domain, we can use the line below instead
domains = UniversalDict(range(1,5))  
           
def h7_constraints(A, a, B, b):
    #A <> B
    if ((A=='A' and B=='B') or (A=='B' and B=='A')): return (a != b)
    
    #E <> C
    if ((A=='E' and B=='C') or (A=='C' and B=='E')): return (a != b) 
    
    #E <> D 
    if ((A=='E' and B=='D') or (A=='D' and B=='E')): return (a != b)
    
    #A < D
    if (A=='A' and B=='D'): return (a < b)
    if (A=='D' and B=='A'): return (a > b)
    
    #D < C
    if (A=='D' and B=='C'): return (a < b)
    if (A=='C' and B=='D'): return (a > b)
    
    #E > B
    if (A=='E' and B=='B'): return (a > b)
    if (A=='B' and B=='E'): return (a < b)    
    
    #E - A is odd and in the domain {1,2,3,4}
    if (A=='E' and B=='A'):
        return ((a-b)==1 or (a-b)==3)
    if (A=='A' and B=='E'):
        return ((b-a)==1 or (b-a)==3)   
    
#______________________________________________________________________________
#Options for CSP:
## select_unassigned_variable: = [first_unassigned_variable,  mrv,  num_legal_values]
## order_domain_values = [unordered_domain_values, lcv]
## inference = [no_inference, forward_checking, mac]

#Test Homework 7
problem = CSP(vars, domains, neighbors, h7_constraints)
result = backtracking_search(problem,select_unassigned_variable=mrv,order_domain_values=lcv,inference=mac) 
print(result)

#Test map coloring problems
#problem = usa  #usa france australia
#result = backtracking_search(problem,select_unassigned_variable=mrv,order_domain_values=lcv,inference=no_inference) #no_inference forward_checking mac
#print(result)

#Test map coloring problems
#problem = NQueen(32)  
#result = backtracking_search(problem,select_unassigned_variable=mrv,order_domain_values=lcv,inference=no_inference) #no_inference forward_checking mac
#print(result)