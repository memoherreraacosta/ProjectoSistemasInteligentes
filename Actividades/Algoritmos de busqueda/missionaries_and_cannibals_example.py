#------------------------------------------------------------------------------------------------------------------
#   Imports
#------------------------------------------------------------------------------------------------------------------

from simpleai.search import SearchProblem, breadth_first, depth_first

#------------------------------------------------------------------------------------------------------------------
#   Helper functions
#------------------------------------------------------------------------------------------------------------------
def list_to_string(input_list):
    """
        This function converts the specified list into a string.

        input_list : The list to be converted.
    """
    return str(input_list[0]) + str(input_list[1]) + str(input_list[2]) + str(input_list[3]) + input_list[4]

def string_to_list(input_string):
    """
        This function converts the specified string into a list.

        input_string : The string to be converted.
    """
    return [int(input_string[0]), int(input_string[1]), int(input_string[2]), int(input_string[3]), input_string[4]]


#------------------------------------------------------------------------------------------------------------------
#   Problem definition
#------------------------------------------------------------------------------------------------------------------

class MissionariesAndCannibals(SearchProblem):
    """ Class that is used to define the missionaries and cannibals problem. """

    def __init__(self):
        """ Class constructor. It initializes the problem with 3 missionaries and 3 cannibals
            at one side of the river. """
        
        initial_state = '3030L'   # Initial state

        # Call base class constructor (the initial state is specified here).
        SearchProblem.__init__(self, initial_state)

        # Define goal state.
        self.goal = '0303R'

    def actions(self, state):
        """ 
            This method returns a list with the possible actions that can be performed according to
            the specified current state.

            state : The game state to evaluate.
        """
        actions = []

        st = string_to_list(state)

        if st[4] == 'L':            
            # One missionary to the other side       
            if st[0] >= 1:
                if ((st[0]-1 >= st[2]) or st[0]-1 == 0) and (st[1]+1 >= st[3]):
                    actions.append('M1R')

            # Two missionaries to the other side       
            if st[0] >= 2:
                if ((st[0]-2 >= st[2]) or st[0]-2 == 0) and (st[1]+2 >= st[3]):
                    actions.append('M2R')

            # One cannibal to the other side       
            if st[2] >= 1:
                if (st[1] >= st[3]+1) or (st[1] == 0):
                    actions.append('C1R')

            # Two cannibals to the other side       
            if st[2] >= 2:
                if (st[1] >= st[3]+2) or (st[1] == 0):
                    actions.append('C2R')

            # One missionary and one cannibal to the other side       
            if st[0] >= 1 and st[2] >= 1:      
                if st[1]+1 >= st[3]+1:
                    actions.append('M1C1R')

        else:
            # One missionary to the other side       
            if st[1] >= 1:
                if ((st[1]-1 >= st[3]) or st[1]-1 == 0) and (st[0]+1 >= st[2]):
                    actions.append('M1L')

            # Two missionaries to the other side       
            if st[1] >= 2:
                if ((st[1]-2 >= st[3]) or st[1]-2 == 0) and (st[0]+2 >= st[2]):
                    actions.append('M2L')

            # One cannibal to the other side       
            if st[3] >= 1:
                if (st[0] >= st[2]+1) or (st[0] == 0):
                    actions.append('C1L')

            # Two cannibals to the other side       
            if st[3] >= 2:
                if (st[0] >= st[2]+2) or (st[0] == 0):
                    actions.append('C2L')

            # One missionary and one cannibal to the other side       
            if st[1] >= 1 and st[3] >= 1:    
                if st[0]+1 >= st[2]+1:
                    actions.append('M1C1L')

        return actions

    def result(self, state, action):
        """ 
            This method returns the new state obtained after performing the specified action.

            state : The game state to be modified.
            action : The action be perform on the specified state.
        """

        st = string_to_list(state)
        m1 = st[0]
        m2 = st[1]    
        c1 = st[2]            
        c2 = st[3]
        b = st[4]

        if action == 'M1R':
            m1-=1;
            m2+=1;
            b='R';
        elif action == 'M2R':
            m1-=2;
            m2+=2;
            b='R';
        elif action == 'C1R':
            c1-=1;
            c2+=1;
            b='R';
        elif action == 'C2R':
            c1-=2;
            c2+=2;
            b='R';
        elif action == 'M1C1R':
            m1-=1;
            m2+=1;
            c1-=1;
            c2+=1;
            b='R';
        elif action == 'M1L':
            m1+=1;
            m2-=1;
            b='L';
        elif action == 'M2L':
            m1+=2;
            m2-=2;
            b='L';
        elif action == 'C1L':
            c1+=1;
            c2-=1;
            b='L';
        elif action == 'C2L':
            c1+=2;
            c2-=2;
            b='L';
        elif action == 'M1C1L':
            m1+=1;
            m2-=1;
            c1+=1;
            c2-=1;
            b='L';
        
        return list_to_string([m1, m2, c1, c2, b])

    def is_goal(self, state):
        """ 
            This method evaluates whether the specified state is the goal state.

            state : The game state to test.
        """
        return state == self.goal

#------------------------------------------------------------------------------------------------------------------
#   Program
#------------------------------------------------------------------------------------------------------------------

# Create solver object\
print("Depth First search")
result = depth_first(MissionariesAndCannibals(), graph_search=True)

# Print results
for i, (action, state) in enumerate(result.path()):
    print()
    if action == None:
        print('Initial configuration')
    elif i == len(result.path()) - 1:
        print('Movement: '+ str(i) +' After moving', action, 'Goal achieved!')
    else:
        print('Movement: '+ str(i) +' After moving', action)

    print(state)

print("------------------------------------------------------")
print("Breadth First search")
result = breadth_first(MissionariesAndCannibals(), graph_search=True)

# Print results
for i, (action, state) in enumerate(result.path()):
    print()
    if action == None:
        print('Initial configuration')
    elif i == len(result.path()) - 1:
        print('Movement: '+ str(i) +' After moving', action, 'Goal achieved!')
    else:
        print('Movement: '+ str(i) +' After moving', action)

    print(state)
#------------------------------------------------------------------------------------------------------------------
#   End of file
#------------------------------------------------------------------------------------------------------------------
