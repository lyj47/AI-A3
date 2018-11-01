from search import Problem
from csp import backtracking_search, no_inference, forward_checking, mac
from csp import first_unassigned_variable,mrv,num_legal_values,unordered_domain_values,lcv
from csp import CSP,usa,france,australia,NQueen,UniversalDict

class Problem:

    board = []
    vars = []

    neighbors = {}
    domains = UniversalDict(range(1))

    def __init__(self, file):
        self.file = file
        self.initialize_board()

    # A = row1, B = row2
    def constraints(A, a, B, b):
        # check row for consecutive 0s

        return True


    def initialize_board(self):

        file_data = open(self.file, 'r+')

        row = 0; col = 0

        board_length = len(file_data.readlines()[0:1][0].strip())

        self.board = [
                        [0 for x in range(board_length)]
                        for y in range(board_length)
                     ]

        # reset the pointer of the file
        file_data.seek(0)

        for line in file_data.readlines():
            col = 0
            for index in range(len(line)-1):
                line = line.strip()
                if line[index] == '-':
                    self.board[row][col] = None
                else:
                    self.board[row][col] = int(line[index])
                col = col + 1
            row = row + 1
        file_data.close()

problem = Problem('p1.txt')
print(problem.board)
problem = CSP(problem.vars, problem.domains, problem.neighbors, problem.constraints)
result = backtracking_search(problem,select_unassigned_variable=mrv,order_domain_values=lcv,inference=mac)
print(result)
