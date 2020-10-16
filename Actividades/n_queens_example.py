#------------------------------------------------------------------------------------------------------------------
#   Simulated annealing solver for the n-queen problem.
#------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------
#   Imports
#------------------------------------------------------------------------------------------------------------------
import time
import random
import math

#------------------------------------------------------------------------------------------------------------------
#   Class definitions
#------------------------------------------------------------------------------------------------------------------

class Board(object):
    """ Class that represents n-queens placed on a chess board. """
    
    def __init__(self, n, randomize = True):        
        """ 
            This constructor initializes the board with n queens.                         

            n : The number of rows and columns of the chess.
            randomize : True indicates that the queen positions are choosen randomly.
                        False indicates that the queen are placed on the first row.
        """
        self.n = n
        self.queens = list()
        if randomize:
            for q in range(n):
                empty_space = False
                while not empty_space:
                    row = random.choice(range(n))
                    col = random.choice(range(n))
                    if not [row, col] in self.queens:
                        empty_space = True
                self.queens.append([row, col])
        else:
            self.queens = [
                [0, q]
                for q in range(n)
            ]

    def show_board(self):        
        """ This method prints the current board. """               
        for row in range(self.n):
            for col in range(self.n):
                if [row, col] in self.queens:
                    print (' Q ', end = '')
                else:
                    print (' - ', end = '')
            print()
        print()
    
    def cost(self):
        """ This method calculates the cost of this solution (the number of queens that are not safe). """
        c = self.n
        for i in range(self.n):
            queen = self.queens[i]
            safe = True
            for j in range(self.n):
                if i == j:
                    continue
                other_queen = self.queens[j]
                if (queen[0] == other_queen[0]):
                    safe = False
                elif (queen[1] == other_queen[1]):
                    safe = False
                elif abs(queen[0]-other_queen[0]) == abs(queen[1]-other_queen[1]):
                    safe = False
            if safe:
                c -= 1
        return c

    def moves(self):
        """ This method returns a list of possible moves given the current placements. """
        move_list = list()
        for i in range(self.n):
            row = self.queens[i][0]
            col = self.queens[i][1]
            for rd in [-1,0,1]:
                for cd in [-1,0,1]:
                    if (rd == 0) and (cd == 0):
                        continue
                    new_pos = [row+rd, col+cd]
                    if (new_pos[0] >= 0) and (new_pos[0] < self.n) and (new_pos[1] >= 0) and (new_pos[1] < self.n):
                        if not new_pos in self.queens: 
                            move_list.append([i, new_pos])

        return move_list

    def neighbor(self):
        """ This method returns a board instance like this one but with one random move made. """        
        newBoard = Board(self.n, False)
        for i in range(self.n):
            newBoard.queens[i][0] = self.queens[i][0]
            newBoard.queens[i][1] = self.queens[i][1]
                    
        current_moves = self.moves()
        n_moves = len(current_moves)
        move_index = random.choice(range(n_moves))
        newBoard.queens[current_moves[move_index][0]] = current_moves[move_index][1]

        return newBoard
    
                                       
#------------------------------------------------------------------------------------------------------------------
#   Program
#------------------------------------------------------------------------------------------------------------------
random.seed(time.time()*1000)

board = Board(8)                # Initialize board
board.show_board()    
cost = board.cost()             # Initial cost    
step = 0;                       # Step count

alpha = 0.9995; # Coefficient of the exponential temperature schedule
t0 = 1;         # Initial temperature
t = t0

while (t > 0.005) and (cost > 0):
    # Calculate temperature
    t = t0 * math.pow(alpha, step)
    step += 1
    # Get random neighbor
    neighbor = board.neighbor()
    new_cost = neighbor.cost()

    # Test neighbor
    if new_cost < cost:
        board = neighbor
        cost = new_cost
    else:
        # Calculate probability of accepting the neighbor
        p = math.exp(-(new_cost - cost)/t)
        if p >= random.random():
            board = neighbor
            cost = new_cost

    print("Iteration: ", step, "    Cost: ", cost, "    Temperature: ", t)

print("--------Solution-----------")
board.show_board()         

#------------------------------------------------------------------------------------------------------------------
#   End of file
#------------------------------------------------------------------------------------------------------------------