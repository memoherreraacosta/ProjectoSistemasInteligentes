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

class Classroom(object):

    groups = [[1,2],[2,3,4],[1,7,3],[10,11,12],[13,14,7,8,9],[0,5,6,15]]

    def __init__(self, n, c, randomize = True):
        """
            This constructor initializes the board with n queens.

            n : The number of rows and columns of the chess.
            randomize : True indicates that the queen positions are choosen randomly.
                        False indicates that the queen are placed on the first row.
        """
        self.n = n
        self.students = []
        self.chocolates = c
        if (randomize):
            self.students = [0 for _ in range(n)]
            for i in range(self.n):
                while self.chocolates>0:
                    pos = random.choice(range(n))
                    if self.students[pos]==0:
                        self.chocolates-=1
                        self.students[pos]=1

        else:
            for i in range(self.n):
                if self.chocolates>0:
                    self.students.append(1)
                    self.chocolates-=1
                else:
                    self.students.append(0)

    # Insatisfaction when a student doesn't get a chocolate
    def get_insatisfaction(self,student): 
        total = 2 # Initial value is 2 due to personal insatisfaction
        friends = []
        for group in self.groups:
            if student in group:
                friends.append(group)
                # print(group)
                total += len(group) - 1
                # print(total)
        # print("Estudiante",student,"sin chocolate. Grupo de amigos:",friends,"insatisfacción =",total)
        # print()
        return total

    def show_classroom(self):
        """ This method prints the current board. """
        print(self.students)

    def cost(self):
        """ This method calculates the cost of this solution (the number of queens that are not safe). """
        # c = self.n
        c = 0
        for i in range(self.n):
            if(not self.students[i]):
                c += self.get_insatisfaction(i)
        # for i in range(self.n):
        #     queen = self.queens[i]
        #     safe = True
        #     for j in range(self.n):
        #         if i == j:
        #             continue
        #         other_queen = self.queens[j]
        #         if (queen[0] == other_queen[0]):
        #             safe = False
        #         elif (queen[1] == other_queen[1]):
        #             safe = False
        #         elif abs(queen[0]-other_queen[0]) == abs(queen[1]-other_queen[1]):
        #             safe = False
        #     if safe:
        #         c -= 1
        # print("Costo de insatisfaccion total:",c)
        return c

    def moves(self):
        """ This method returns a list of possible moves given the current placements. """
        available_spaces = [pos for pos, x in enumerate(self.students) if x == 0]

        move_list = []
        for i in range(self.n):
            student = self.students[i]
            if student == 1 and available_spaces:
                spaces = len(available_spaces)
                move_index = random.choice(range(spaces))
                j = available_spaces.pop(move_index)
                tmp = self.students.copy()
                tmp[i] = 0
                tmp[j] = 1
                move_list.append(tmp)
        print(move_list)
        return move_list

    def neighbor(self):
        """ This method returns a board instance like this one but with one random move made. """
        newClass = Classroom(self.n, False)

        current_moves = self.moves()
        n_moves = len(current_moves)
        move_index = random.choice(range(n_moves))
        newClass.students = current_moves[move_index]
        print(newClass.students)
        return newClass


#------------------------------------------------------------------------------------------------------------------
#   Program
#------------------------------------------------------------------------------------------------------------------
# random.seed(time.time()*1000)

classroom = Classroom(16, 8)                # Initialize board
# print("Estudiantes seleccionados:",classroom.students)
print()
classroom.show_classroom()
cost = classroom.cost()             # Initial cost
step = 0;                       # Step count

alpha = 0.9995; # Coefficient of the exponential temperature schedule
t0 = 1;         # Initial temperature
t = t0

while (t > 0.005):

    # Calculate temperature
    t = t0 * math.pow(alpha, step)
    step += 1

    # Get random neighbor
    neighbor = classroom.neighbor()
    new_cost = neighbor.cost()

    # Test neighbor
    if new_cost < cost:
        classroom = neighbor
        cost = new_cost
    else:
        # Calculate probability of accepting the neighbor
        p = math.exp(-(new_cost - cost)/t)
        if p >= random.random():
            classroom = neighbor
            cost = new_cost

    print("Iteration: ", step, "    Cost: ", cost, "    Temperature: ", t)

print("--------Solution-----------")
classroom.show_classroom()

#------------------------------------------------------------------------------------------------------------------
#   End of file
#------------------------------------------------------------------------------------------------------------------
