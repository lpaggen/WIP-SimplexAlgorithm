# import the necessary libraries, in this case NumPy for its arrays
import numpy as np

# creating a class which can instantiate objects
class SimplexTools():
    """Class containing optimization-problem solving algorithms"""
    obj_fnc = [0.0] # starts with a 0 to account for the constant in the constraints while tranforming problem to dictionary format
    constraints = [] # empty list of constraints, used to create a system of equations later on
    dictionaries = [] # list of dictionaries, essentially what every other method operates on and modifies
    def __init__(self, n, m): # will potentially add an option to have the problem as a minimization or a maximization problem
        self.var_count = n # number of variables to the problem
        self.constr_count = m # number of bounds to the problem

    # implementing a way to store the information of a LP problem (vaiables, constraints)
    def _formulate_problem(self): # n variables, m constraints
        """Takes n, m as arguments: n = variables, m = constraints"""
        
        # creating the objective function of an LP problem
        for i in range(0, self.var_count): # ranges from 0 to the number of variables in the problem
            try:
                obj_coeff = float(input(f"OBJECTIVE FUNCTION -> Coefficient of X_{i} = ")) # assigning n coefficients to n variables
            except ValueError:
                print("invalid input, only floats and integers are accepted")
            self.obj_fnc.append(obj_coeff) # appends each coefficient to later convert to an array
            
        # creating each constraint of the LP problem
        for i in range(0, self.constr_count): # determines how many constraints to create
            constraint_m = ["RHS"] # named constraint_m to represent the "M'th" constraint of the problem
            j = 1
            while j < self.var_count + 1:
                try:
                    constr_coeff = -float(input(f"CONSTRAINT_{i + 1} -> Coefficient of X_{j - 1} = ")) # assigning n coefficients to n variables
                except ValueError:
                    print("Invalid input, only floats and integers are accepted")
                    self._formulate_problem() # trying recursion, don't know if necessary at all
                constraint_m.append(constr_coeff) # appends each constraint coefficient to a list
                j += 1 # incrementing j to n + 1, at which point we exit this loop
            try: # exited the loop, no need for "else" statement
                constr_rhs = float(input(f"RHS of constraint #{i + 1} = "))
            except ValueError:
                print("Invalid input, only floats and integers are accepted")
                self._formulate_problem # trying recursion, don't know if necessary at all
            constraint_m[0] = constr_rhs # appends the RHS of the constraint to the same list
            constraint_m = np.array(constraint_m)
            self.constraints.append(constraint_m) # appends the list (which is now constraint coeffs + RHS) to a list of constraints
        self.obj_fnc = np.array(self.obj_fnc) # turns list (objective function) into an array

    def _to_dict(self):
        """Converts the objective function and relevant constraints into a dictionary"""
        A = np.vstack((self.obj_fnc, self.constraints)) # more efficient than appending and resizing
        slack_vector = np.vstack(("Z_", np.ones((self.constr_count, 1)))) # same as above, Z can be anything
        empty_slack_pivot_matrix = np.zeros((self.constr_count + 1, self.constr_count)) # empty matrix to append to the dictionary
        dictionary = np.hstack((slack_vector, A, empty_slack_pivot_matrix)) # creates the dictionary
        self.dictionaries.append(dictionary)

    # def display_dict(self):
    #     """returns the dictionary of the relevant problem"""
    #     var_indicators = ["Z/W"]
    #     for i in range(0, self.var_count):
    #         var_indicators.append(f"X_{i}")
    #     for i in range(0, self.constr_count):
    #         var_indicators.append(f"W_{i + 1}")
    #     var_indicators = np.array(var_indicators)
    #     for i in SimplexTools.dictionaries:
    #         SimplexTools.dictionaries[i] = np.vstack((var_indicators, SimplexTools.dictionaries[i]))
    #     return "yes?"

    def _pivot(self):
        """Identifies the column and row to pivot for which a variable will enter the basis"""
        # setting up the basic parameters of the function
        pivot_iteration = 1 # general pivot iteration, only used for descriptive statistics at the end
        dictionary = self.dictionaries[0] # fetches the problem as transformed by "_formulate_problem()"
        print(dictionary) # debug, works as intented
        
        # dictionary which keeps track of the current iteration of each constraint (used mainly for 1st iteration)
        iteration_tracker = {}
        for i in range(0, self.constr_count): # starts at 0, indexing similar to normal lists etc
            constr = f"{i}" # makes future statements easier, rather than using some complex naming
            iteration_tracker[constr] = 1 # sets the value of each constraint iteration count to 1
            print(iteration_tracker)

        # find the variable with highest coeff in obj function
        pivot_col_index = np.argmax(dictionary[0, 1:]) + 1 # Z column is excluded, add 1
        pivot_col = dictionary[1:, pivot_col_index].astype(np.float64)

        # determine the variables which enter/leave the basis
        RHS_col = dictionary[1:, 1].astype(np.float64)
        try:
            ratio = pivot_col / RHS_col
            pivot_row_index = np.argmin(ratio) + 1
        except ZeroDivisionError:
            pass
        entering_var = dictionary[pivot_row_index, pivot_col_index] # + 1 account for obj function
        exiting_var = dictionary[pivot_row_index, 0]

        # finding the coordinates ij of entering and exiting variables before the pivot
        entering_var_coord = pivot_row_index, pivot_col_index # this is a tuple of 2 coordinates
        print(entering_var_coord) # debug
        exiting_var_coord = pivot_row_index, 0 # tuple of 2 coordinates
        print(exiting_var_coord) # debug
        iteration_tracker_index = str(exiting_var_coord[0] - 1)
        print(iteration_tracker_index) # debug 

        # assigning the coordinates ij of entering and exiting variables after the pivot
        if iteration_tracker.get(iteration_tracker_index) == 1: # checks if the iteration of the pivot row is 1
            exiting_var_coord_after_pivot = pivot_row_index, (2 + self.var_count + pivot_row_index)
            print(exiting_var_coord_after_pivot) # debug
        else:
            exiting_var_coord_after_pivot = entering_var_coord # no, this will not work as expected, it needs to be entering_var_coord(iteration-1) somehow

    def solve(self):
        """Solves an LP maximization problem and gives optimal solutions to the problem"""
        # set entering variable for the _pivot method
