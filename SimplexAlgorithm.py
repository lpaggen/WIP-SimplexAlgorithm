# import the necessary libraries, in this case NumPy for its arrays
import numpy as np

# creating a class which can instantiate objects
class SimplexTools():
    """Class containing optimization-problem solving algorithms"""

    def __init__(self, n, m): # will potentially add an option to have the problem as a minimization or a maximization problem
        self.var_count = n # number of variables to the problem
        self.constr_count = m # number of bounds to the problem
        self.obj_fnc = [0.0] # 0 to account for the constant in the constraints while tranforming problem to dictionary format
        self.constraints = [] # empty list of constraints, used to create a system of equations later on
        self.dictionaries = [] # list of dictionaries, essentially what every other method operates on and modifies
        self.pivot_iteration = 0 # keeps track of iterations at a certain point t
        self.iter_tracker = {} # dictionary tracking iterations per constraint
        self.enter_coord_dict = {}
        self.prev_enter_coordinates = [] # sublist
        self.basis_var = {}
        self.var_indicators = ["bss", "cst"] # basis, constant

        # dict to keep track of iterations PER constraint and positions of basis variables
        for i in range(0, self.constr_count): # starts at 0, indexing similar to normal list
            constr = f"{i}" # makes future statements easier, rather than using some complex naming
            self.iter_tracker[constr] = 1 # sets the value of each constraint iteration count to 1
            self.enter_coord_dict[constr] = []
            self.basis_var[constr] = []

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
                self._formulate_problem() # trying recursion, don't know if necessary at all
            constraint_m[0] = constr_rhs # appends the RHS of the constraint to the same list
            constraint_m = np.array(constraint_m)
            self.constraints.append(constraint_m) # appends the list (which is now constraint coeffs + RHS) to a list of constraints
        self.obj_fnc = np.array(self.obj_fnc) # turns list (objective function) into an array

    def _to_dict(self):
        """Converts the objective function and relevant constraints into a dictionary"""
        A = np.vstack((self.obj_fnc, self.constraints)) # more efficient than appending and resizing
        slack_vector = np.vstack((0, np.ones((self.constr_count, 1)))) # same as above, Z can be anything
        empty_slack_pivot_matrix = np.zeros((self.constr_count + 1, self.constr_count)) # empty matrix to append to the dictionary
        dictionary = np.hstack((slack_vector, A, empty_slack_pivot_matrix)) # creates the dictionary
        self.dictionaries.append(dictionary)

    def _pivot(self):
        """Identifies the column and row to pivot for which a variable will enter the basis"""
        # setting up the basic parameters of the function
        dictionary = self.dictionaries[0] # fetches the problem as transformed by "_formulate_problem()"
        print(dictionary) # debug, works as intented

        # find the variable with highest coeff in obj function
        pivot_col_index = np.argmax(dictionary[0, 2:]) + 2 # Z AND optimal value excl.
        pivot_col_no_obj = dictionary[1:, pivot_col_index].astype(np.float64)
        pivot_col = dictionary[:, pivot_col_index].astype(np.float64)

        # determine the variables which enter/leave the basis
        RHS_col = dictionary[1:, 1].astype(np.float64)
        try:
            ratio = pivot_col_no_obj / RHS_col
            pivot_row_index = np.argmin(ratio) + 1
            pivot_row = dictionary[pivot_row_index, :]
        except ZeroDivisionError:
            pass
        entering_var = dictionary[pivot_row_index, pivot_col_index] # + 1 account for obj function
        exiting_var = dictionary[pivot_row_index, 0]

        # finding the coordinates ij of entering and exiting variables before the pivot
        entering_var_coord = pivot_row_index, pivot_col_index # this is a tuple of 2 coordinates
        print(f"entering_var: {entering_var_coord}") # debug
        exiting_var_coord = pivot_row_index, 0 # tuple of 2 coordinates
        print(f"exiting_var: {exiting_var_coord}") # debug
        iteration_tracker_index = str(exiting_var_coord[0] - 1)
        print(iteration_tracker_index) # debug

        # coordinates of entering var before pivot get stored in dictionary
        constr_index = str(entering_var_coord[0] - 1)
        self.enter_coord_dict[constr_index].append(entering_var_coord)

        # assign the coordinates ij of entering and exiting variables after the pivot
        if self.iter_tracker.get(iteration_tracker_index) == 1: # checks if the iteration of the pivot row is 1
            exiting_var_coord_after_pivot = pivot_row_index, (int(iteration_tracker_index) + self.var_count + 2) #temporary
            print(f"exiting var goes to -> {exiting_var_coord_after_pivot}") # debug
            self.iter_tracker[iteration_tracker_index] += 1 # increment pivot iter by 1
            print("this works")
        else:
            exiting_var_coord_after_pivot = self.enter_coord_dict[constr_index][0] # FIX: iteration_index
            self.enter_coord_dict[constr_index].pop(0) # remove index 0, only need 2 elements at iter t
            print(f"all entering var positions -> {self.prev_enter_coordinates}")
            print(f"exiting var goes to -> {exiting_var_coord_after_pivot}")

        # exiting variable pivots out of the basis
        dictionary[exiting_var_coord_after_pivot] = -exiting_var # change coefficient sign
        dictionary[exiting_var_coord] = 0.0 # 0 assigned to where exiting var was pre pivot

        # multiplication ratio by which to add pivot row to other rows (incl obj row)
        pivot_col_no_enter_var_coords = []
        for i in range(self.constr_count + 1):
            if i != pivot_row_index:
                coords = i, pivot_col_index
                pivot_col_no_enter_var_coords.append(coords)
        print(f"pivot row no enter var -> {pivot_col_no_enter_var_coords}")
        pivot_col_no_enter_var = [dictionary[i] for i in pivot_col_no_enter_var_coords]
        mult_ratio = pivot_col_no_enter_var / entering_var
        print(f"Multiplication Ratio -> {mult_ratio}")

        # get rid of instances of entering variable in other rows
        tolerance = 1e-10 # small tolerance for comparing floats
        mult_ratio_index = 0
        for row in range(0, self.constr_count + 1):
            row_to_modify = np.all(np.isclose(dictionary[row, :], pivot_row, atol = tolerance))
            if not row_to_modify:
                dictionary[row, :] = dictionary[row, :] - (mult_ratio[mult_ratio_index] * pivot_row)
                mult_ratio_index = (mult_ratio_index + 1) % len(mult_ratio)
            else:
                pass
        print(f"mult ratio index -> {mult_ratio_index}")

        # entering variable pivots into the basis
        dictionary[exiting_var_coord] = -entering_var # change signs
        dictionary[entering_var_coord] = 0.0
        dictionary[pivot_row_index, :] = dictionary[pivot_row_index, :] / dictionary[exiting_var_coord]
        print("\nAfter pivoting:\n")
        print(dictionary)
        
        # TO DO: identify which variable is in the basis!!!!!!

        # update the dictionary in the class list
        self.dictionaries[0] = dictionary

        # increment pivot by 1
        self.pivot_iteration += 1
        
    def _display_dict(self):
        """returns the dictionary of the relevant problem"""
        for i in range(0, self.var_count):
            self.var_indicators.append(f"X_{i}")
        for i in range(0, self.constr_count):
            self.var_indicators.append(f"W_{i + 1}")
        var_indicators = np.array(self.var_indicators)
        self.dictionaries[0] = np.vstack((var_indicators, self.dictionaries[0]))

def solve(n, m):
    """
    Returns the solutions to an LP maximization problem
    
    Parameters: n = number of variables in the problem
                m = number of constraints in the problem
    """
    problem = SimplexTools(n, m)
    problem._formulate_problem()
    problem._to_dict()
    tableau = problem.dictionaries[0]
    obj_fnc = tableau[0, 2:]
    while np.any(obj_fnc > 0):
        problem._pivot()
    problem._display_dict()
    print(tableau)
    print("problem solved")

def main(n,m):
    simplex = SimplexTools(n,m)
    simplex._formulate_problem()
    simplex._to_dict()
    simplex._pivot()

if __name__ == "__main__":
    main(2,3)