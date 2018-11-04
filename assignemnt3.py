from search import Problem
from csp import backtracking_search, no_inference, forward_checking, mac
from csp import first_unassigned_variable,mrv,num_legal_values,unordered_domain_values,lcv
from csp import CSP,usa,france,australia,NQueen,UniversalDict
import itertools

class Problem:

    board = []
    vars = []

    neighbors = {}
    domains = {}

    board_length = 0

    def __init__(self, file):
        self.file = file
        self.initialize_values()

    def constraints(self, A, a, B, b):
        if (A < len(a) and B < len(a)) or(A>=len(a) and B>=len(a)):
            if a == b:
                return False
        else:
            if A < len(a):
                if a[B-len(a)] != b[A]:
                    return False

            if B < len(a):
                if a[B] != b[A-len(a)]:
                    return False

        if(A < len(self.board)):
            graph = self.board[A]
            for i in range(0, len(a)):
                if graph[i] != "-" and graph[i] != a[i]:
                    return False

        if(B < len(self.board)):
            graph = self.board[B]
            for i in range(0, len(b)):
                if graph[i] != "-" and graph[i] != b[i]:
                    return False

        return True

    def initialize_values(self):

        file_data = open(self.file, 'r+')

        # get board length
        self.board_length = len(file_data.readlines()[0:1][0].strip())

        # initialize vars, board, and neighbors
        # if the number is greater than the board_length, then it's a col
        self.vars = [x for x in range(0, self.board_length*2)]

        # upper_bound is the max neighbor
        upper_bound = len(self.vars)

        # generate the neighbors ( 1 - N )
        for row in range(0, upper_bound):
            self.neighbors[row] = []
            for other_row in range(0, upper_bound):
                if row != other_row:
                    self.neighbors[row].append(other_row)

        # reset the pointer of the file
        file_data.seek(0)

        # fill in the board
        for line in file_data.readlines():
            row = list(line)
            if '\n' in row:
                row.remove('\n')
            self.board.append(row)
        file_data.close()

        # the domain to find permutations of
        permutation_domain = []
        # the permutations found
        permutations = []
        # permutations that match the 3 non-consecutive 0s or 1s
        valid_permutations = []

        # generate the permutation_domain
        for index in range(self.board_length):
            if index < self.board_length/2:
                permutation_domain.append('0')
            else:
                permutation_domain.append('1')

        # generate all possible permutations
        for perm in list(set(itertools.permutations(permutation_domain, self.board_length))):
            permutations.append(list(perm))

        # generate all valid_permutations
        for permutation_index in range(len(permutations)):
            is_valid = True
            for index in range(len(permutations[permutation_index])-2):
                permutation = permutations[permutation_index]
                if permutation[index] == '0' and permutation[index+1] == '0' and permutation[index+2] == '0':
                    is_valid = False
                    break
                if permutation[index] == '1' and permutation[index+1] == '1' and permutation[index+2] == '1':
                    is_valid = False
                    break
            if is_valid:
                valid_permutations.append(permutations[permutation_index])

        self.domains = UniversalDict(valid_permutations)

def main():
    problem = Problem('p2.txt')
    csp = CSP(problem.vars, problem.domains, problem.neighbors, problem.constraints)
    result = backtracking_search(csp,select_unassigned_variable=mrv,order_domain_values=lcv,inference=mac)
    for result_index in range(problem.board_length):
        row = ''
        for value_index in range(len(result[result_index])):
            row = row + str(result[result_index][value_index])
        print(row)

main()
