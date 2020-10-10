#------------------------------------------------------------------------------------------------------------------
#   8-puzzle solver using the A* algorithm.
#
#   This code is an adaptation of the 8-puzzle solver described in:
#   Artificial intelligence with Python. Alberto Artasanchez and Prateek Joshi. 2nd edition, 2020, 
#   editorial Pack. Chapter 10.
#
#------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------
#   Imports
#------------------------------------------------------------------------------------------------------------------

from simpleai.search import astar, SearchProblem
import random

#------------------------------------------------------------------------------------------------------------------
#   Auxiliar functions
#------------------------------------------------------------------------------------------------------------------

def list_to_string(input_list):
    """
        This function converts the specified list into a string.

        input_list : The list to be converted.
    """
    return '\n'.join(['-'.join(x) for x in input_list])

def string_to_list(input_string):
    """
        This function converts the specified string into a list.

        input_string : The string to be converted.
    """
    return [x.split('-') for x in input_string.split('\n')]

def get_location(rows, input_element):
    """
        This function finds the 2D location of the specified input element in a 2-D list.

        rows : A 2-D list that represents a game state.
        input_element : The element to find in the list.
    """
    for i, row in enumerate(rows):
        for j, item in enumerate(row):
            if item == input_element:
                return i, j  

def randomMovements(rows, n):
    """
        This function performs n random movements.

        rows : A 2-D list that represents the board state to modify.
        n : The number of random movements.
    """
    board = rows
    e_row, e_col = get_location(board, 'e')
    board_size = len(board)    

    for i in range(n):
        mov_ok = False
        while not mov_ok:
            mov = random.randint(1,4)
            if mov == 1 and e_row > 0:
                board[e_row][e_col], board[e_row-1][e_col] = board[e_row-1][e_col], board[e_row][e_col]
                e_row-=1
                mov_ok = True
            elif mov == 2 and e_row < board_size-1:    
                board[e_row][e_col], board[e_row+1][e_col] = board[e_row+1][e_col], board[e_row][e_col]
                e_row+=1
                mov_ok = True
            elif mov == 3 and e_col > 0:
                board[e_row][e_col], board[e_row][e_col-1] = board[e_row][e_col-1], board[e_row][e_col]
                e_col-=1
                mov_ok = True
            elif mov == 4 and e_col < board_size-1: 
                board[e_row][e_col], board[e_row][e_col+1] = board[e_row][e_col+1], board[e_row][e_col]
                e_col+=1
                mov_ok = True
    return board;

#------------------------------------------------------------------------------------------------------------------
#   Problem definition
#------------------------------------------------------------------------------------------------------------------

class EightPuzzleProblem(SearchProblem):
    """ Class that is used to obtain the actions and results of the 8-puzzle game. """
    
    def __init__(self, initial_state):
        """ This constructor initializes the 8-puzzle game problem. """
        
        # Call base class constructor (the initial state is specified here).
        SearchProblem.__init__(self, initial_state)

        # Store game goal.
        self.goal = 'e-1-2\n3-4-5\n6-7-8';

        # Create a cache for the goal position of each piece
        self.goal_positions = {}
        self.rows_goal = string_to_list(self.goal)
        for number in 'e12345678':
            self.goal_positions[number] = get_location(self.rows_goal, number)


    def actions(self, state):
        """ 
            This method returns a list with the possible actions that can be performed according to
            the specified current state.

            state : The game state to evaluate.
        """

        rows = string_to_list(state)
        row_empty, col_empty = get_location(rows, 'e')

        actions = []
        if row_empty > 0:
            actions.append(rows[row_empty - 1][col_empty])
        if row_empty < 2:
            actions.append(rows[row_empty + 1][col_empty])
        if col_empty > 0:
            actions.append(rows[row_empty][col_empty - 1])
        if col_empty < 2:
            actions.append(rows[row_empty][col_empty + 1])

        return actions
        
    def result(self, state, action):
        """ 
            This method returns the new state obtained after performing the specified action.

            state : The game state to be modified.
            action : The action be perform on the specified state.
        """
        rows = string_to_list(state)
        row_empty, col_empty = get_location(rows, 'e')
        row_new, col_new = get_location(rows, action)

        rows[row_empty][col_empty], rows[row_new][col_new] = \
                rows[row_new][col_new], rows[row_empty][col_empty]

        return list_to_string(rows)
        
    def is_goal(self, state):
        """ 
            This method evaluates whether the specified state is the goal state.

            state : The game state to test.
        """
        return state == self.goal

    def heuristic(self, state):
        """ 
            This method returns an estimate of the distance from the specified state to 
            the goal using the manhattan distance.

            state : The game state to evaluate.
        """

        rows = string_to_list(state)

        distance = 0

        for number in 'e12345678':
            row_current, col_current = get_location(rows, number)
            row_goal, col_goal = self.goal_positions[number]

            ############## Heuristic function 1
            # Is the element in the right position?
            #distance += int(row_current != row_goal or col_current != col_goal)
            ##############

            ############## Heuristic function 2
            # Distance between the goal position and the current position
            distance += abs(row_current - row_goal) + abs(col_current - col_goal)
            ##############

        return distance

#------------------------------------------------------------------------------------------------------------------
#   Program
#------------------------------------------------------------------------------------------------------------------

# Initialize board
initial_board = randomMovements(string_to_list('e-1-2\n3-4-5\n6-7-8'), 1000)
initial_state = list_to_string(initial_board)

# Create solver object
result = astar(EightPuzzleProblem(initial_state), graph_search=True)

# Print results
for i, (action, state) in enumerate(result.path()):
    print()
    if action == None:
        print('Initial configuration')
    elif i == len(result.path()) - 1:
        print('After moving', action, 'into the empty space. Goal achieved!')
    else:
        print('After moving', action, 'into the empty space')

    print(state)

#------------------------------------------------------------------------------------------------------------------
#   End of file
#------------------------------------------------------------------------------------------------------------------
