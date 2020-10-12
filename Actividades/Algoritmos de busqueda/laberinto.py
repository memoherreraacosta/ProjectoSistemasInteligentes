import math
from simpleai.search import SearchProblem, astar

#------------------------------------------------------------------------------------------------------------------
#   Problem definition
#------------------------------------------------------------------------------------------------------------------

# Class containing the methods to solve the maze
class MazeSolver(SearchProblem):
    # Initialize the class 
    def __init__(self, board):
        self.board = board
        self.goal = (0, 0)
        for y in range(len(self.board)):
            for x in range(len(self.board[y])):
                if self.board[y][x] == "O":
                    self.initial = (x, y)
                elif self.board[y][x] == "X":
                    self.goal = (x, y)
        super(MazeSolver, self).__init__(initial_state=self.initial)

    # Define the method that takes actions
    # to arrive at the solution
    def actions(self, state):
        actions = []
        for action in COSTS.keys():
            newx, newy = self.result(state, action)
            if self.board[newy][newx] != "+":
                actions.append(action)
        return actions

    # Update the state based on the action
    def result(self, state, action):
        x, y = state
        if action.count("up"):
            y -= 1
        if action.count("down"):
            y += 1
        if action.count("left"):
            x -= 1
        if action.count("right"):
            x += 1
        new_state = (x, y)
        return new_state

    # Check if we have reached the goal
    def is_goal(self, state):
        return state == self.goal

    # Compute the cost of taking an action
    def cost(self, state, action, state2):
        return COSTS[action]

    # Heuristic that we use to arrive at the solution
    def heuristic(self, state):
        x, y = state
        gx, gy = self.goal
        return math.sqrt((x - gx) ** 2 + (y - gy) ** 2)

if __name__ == "__main__":
    # Define the map
    MAP = ["""
++++++++++++++++++++++
+ O +   ++ ++        +
+     +     +++++++ ++
+ +    ++  ++++ +++ ++
+ +   + + ++         +
+          ++  ++  + +
+++++ + +      ++  + +
+++++ +++  + +  ++   +
+          + +  + +  +
+++++ +  + + +     X +
++++++++++++++++++++++
    """

    
    ,"""
++++++++++++++++++++++++++++++
+ O       +              +   +
+ ++++    ++++++++       +   +
+    +    +              +   +
+    +++     +++++  ++++++   +
+      +   +++   +           +
+      +     +   +  +  +   +++
+     +++++    +    +  + X   +
+              +       +     +
++++++++++++++++++++++++++++++
    """

 
    ,"""
++++++++++++++++++++++++++++++
+         +              +   +
+ ++++    ++++++++       +   +
+  O +    +              +   +
+    +++     +++++  ++++++   +
+      +   +++   +           +
+      +     +   +  +  +   +++
+     +++++    +    +  + X   +
+              +       +     +
++++++++++++++++++++++++++++++
    """

    ,"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+O                       +                  +      +     + + +  +  +   +++++++++++++++  ++
+  +++    ++++++++       +                  +      +     + + +  +  +   +++++++++++++++  ++
+  + +    +              +                  +      +     + + +  +  +   +++++++++++++++  ++
+    +++     +++++ +++++++                  +      +     + + +  +  +   +++++++++++++++  ++
+      +   ++++                             +      +     + + +  +  +   +++++++++++++++  ++
+      +     + + +  +  +   +++++++++++++++  ++  ++          + +           +              +
+     +++++        ++  + + + +              ++  ++          + +           +              +
+  ++          + +           +              ++  ++          + +           +              +
+ + + + + + + + + + + + +  +++++++++++++++ ++ + + + + + + + + + + + +  +++++++++++++++ +++
+  ++          + +           ++++++++++++ +++ + + + + + + + + + + + +  +++++++++++++++ +++
+  ++          + +           +++++++++++++  + + + + + + + + + + + + +  +++++++++++++++ +++
+  ++          + +           +              ++  ++          + +           +              +
+  ++          + +           +              ++  ++          + +           +              +
+  ++          + +           +              ++  ++          + +           +              +
+  ++          + +           +              ++  ++          + +           +              +
+  ++          + +           +              ++  ++          + +           +              +
+  ++          + +           +              ++  ++          + +           +              +
+  +++    ++++++++       +                  ++  ++          + +           +              +
+  + +    +              +                  ++  ++          + +           +              +
+    +++     +++++ +++++++                         +    +++     +++++ +++++++  +++++++++++
+      +   ++++                             ++  ++          + +           +              +
+      +     + + +  +  +   +++++++++++++++  ++  ++          + +           +              +
+     +++++        ++  + + + +              ++  ++          + +           +              +
+  ++          + +           +              ++  ++          + +           +              +
+ + + + + + + + + + + + +  +++++++++++++++ +++  ++          + +           +              +
+  ++          + +           ++++++++++++ ++++  ++          + +           +              +
+  ++          + +           +++++++++++++  ++  ++          + +           +              +
+  ++          + +           +              ++  ++          + +           +              +
+  ++          + +           +              ++  ++          + +           +              +
+  ++          + +           +              ++  ++          + +           +              +
+  ++          + +           +              ++  ++          + +           +              +
+  ++          + +           +              ++  ++          + +           +              +
+  ++          + +           +              ++  ++          + +           +              +
+  +++    ++++++++       +                  ++  ++          + +           +              +
+  + +    +              +                  ++  ++          + +           +              +
+    +++     +++++ +++++++                  ++  ++          + +           +              +
+      +   ++++                             ++  ++          + +           +              +
+      +     + + +  +  +   +++++++++++++++  ++    +++     +++++ +++++++                  +
+     +++++        ++  + + + +              ++    +++     +++++ +++++++                  +
+  ++          + +           +              ++ + + + + + + + + + + + +  +++++++++++++++ ++
+ + + + + + + + + + + + +  +++++++++++++++ +++ + + + + + + + + + + + +  +++++++++++++++ ++
+  ++          + +           ++++++++++++ ++++ + + + + + + + + + + + +  +++++++++++++++ ++
+  ++          + +           +++++++++++++  ++  ++          + +           +++++++++++++  +
+  ++          + +           +              ++  ++          + +           +++++++++++++  +
+  ++          + +           +              ++  ++          + +           +++++++++++++  +
+  ++          + +           +              ++  ++          + +           +++++++++++++  +
+  ++          + +           +              ++  ++          + +           +++++++++++++  +
+  ++          + +           +              ++  ++          + +           +++++++++++++  +
+  ++          + +           +              ++  ++          + +           +++++++++++++  +
+  ++          + +           +              ++  ++          + +     X     +++++++++++++  +
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """]

    for map in MAP: 
        # Convert map to a list
        print("------MAP------")
        print(map)
        map = [list(x) for x in map.split("\n") if x]

        # Define cost of moving around the map
        cost_regular = 1.0
        cost_diagonal = 1.7

        # Create the cost dictionary
        COSTS = {
            "up": cost_regular,
            "down": cost_regular,
            "left": cost_regular,
            "right": cost_regular,
            "up left": cost_diagonal,
            "up right": cost_diagonal,
            "down left": cost_diagonal,
            "down right": cost_diagonal,
        }

        # Create maze solver object
        problem = MazeSolver(map)

        # Run the solver
        result = astar(problem, graph_search=True)

        # Extract the path
        path = [x[1] for x in result.path()]
        print("--Solution--")
        print()
        # Print the steps and coordinate path as string
        print(result.path())
        # Print the map
        print()
        for y in range(len(map)):
            for x in range(len(map[y])):
                if (x, y) == problem.initial:
                    print('O', end='')
                elif (x, y) == problem.goal:
                    print('X', end='')
                elif (x, y) in path:
                    print('*', end='')
                else:
                    print(map[y][x], end='')

            print()