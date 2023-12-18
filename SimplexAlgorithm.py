# import the necessary libraries, in this case NumPy for its arrays
import numpy as np

# creating a class which can instantiate objects
class SimplexTools():
    """Class containing optimization-problem solving algorithms"""

    def __init__(self, obj, constraint_list, n, m): # trying something new
        self.obj = obj
        self.constraint_list = constraint_list
        self.var_count = n # number of variables to the problem
        self.constr_count = m # number of bounds to the problem
        self.obj_fnc = [0.0] # 0 to account for the constant in the constraints while tranforming problem to dictionary format
        self.constraints = [] # list of constraints, used to create a system of equations later on
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
        """creates usable constraints from user input in solve function"""
        # create objective function
        print("creating obj")
        self.obj_fnc = [0] + self.obj[0:]
        self.obj_fnc = np.array(self.obj_fnc).astype(np.float64)
        
        # create constraints
        print("creating cst")
        for constraint in self.constraint_list:
            cst = [constraint[-1]] + [-i for i in constraint if i != constraint[-1]] # individual constraint is a list
            self.constraints.append(cst)

        # debugging purposes
        print(self.obj_fnc)
        print(self.constraints)

    def _to_dict(self):
        """Converts the objective function and relevant constraints into a dictionary"""
        A = np.vstack((self.obj_fnc, self.constraints)) # more efficient than appending and resizing
        slack_vector = np.vstack((0, np.ones((self.constr_count, 1)))) # same as above, Z can be anything
        empty_slack_pivot_matrix = np.zeros((self.constr_count + 1, self.constr_count)) # empty matrix to append to the dictionary
        dictionary = np.hstack((slack_vector, A, empty_slack_pivot_matrix)) # finalizes the dictionary
        self.dictionaries.append(dictionary)

    def _pivot(self):
        """Identifies the column and row to pivot for which a variable will enter the basis"""
        # setting up the basic parameters of the function
        dictionary = self.dictionaries[0] # fetches the problem as transformed by "_formulate_problem()"

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
        entering_var = dictionary[pivot_row_index, pivot_col_index]
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
        
        # test
        print(self.iter_tracker)
        
    def _display_dict(self):
        """returns the dictionary of the relevant problem"""
        for i in range(0, self.var_count):
            self.var_indicators.append(f"X_{i}")
        for i in range(0, self.constr_count):
            self.var_indicators.append(f"W_{i + 1}")
        var_indicators = np.array(self.var_indicators)
        self.dictionaries[0] = np.vstack((var_indicators, self.dictionaries[0]))
        
        # TO DO
        # SOLVE FOR OPTIMAL VALUES OF EQUATIONS (solve for non basic = 0)
        # have to save dictionary of iteration -1 in order to do this 
        # in dictionary t-1, need to set all non basic = 0
        # solve for all basics, report in final solution

def solve(obj, constraints):
    """
    Takes a Linear Programming (LP) maximization problem and returns its optimal dictionary.

    Parameters:
    - obj: List or array representing the coefficients of the objective function.
    - constraints: List or array containing the constraints of the LP problem.

    Returns:
    A dictionary containing the solved values for the variables in the objective function, 
    maximizing the given objective function within the specified constraints.
    
    Examples:
    >>> # Define an objective function and constraints
    >>> objective_function = [3, 4, 5]  # Objective function: 3x + 4y + 5z
    >>> problem_constraints = [
    >>>     [1, 2, 4, 5],  # Constraint 1: x + 2y <= 5
    >>>     [4, -1, -3, 8],  # Constraint 2: 4x - y -3z <= 8
    >>>     [2, 0, 1, 3],  # Constraint 3: 2x + z <= 3
    >>> ]

    >>> # Solve the LP problem
    >>> solution = solve(objective_function, problem_constraints)
    >>> print(solution)
    [[ 0.   11.5   0.    0.   -3.5  -2.    0.   -0.5 ]
     [ 1.    1.75  0.    0.   -1.75 -0.5   0.    0.25]
     [ 1.    3.75  0.    0.    3.25 -0.5   0.    2.25]
     [ 1.    1.5   0.    0.   -0.5   0.    0.   -0.5 ]]
    """
    m = len(constraints)
    n = len(constraints[0]) - 1
    problem = SimplexTools(obj, constraints, n, m)
    problem._formulate_problem()
    problem._to_dict()
    tableau = problem.dictionaries[0]
    obj_fnc = tableau[0, 2:]
    while np.any(obj_fnc > 0):
        problem._pivot()
    problem._display_dict()
    print(tableau)
    print("problem solved")
