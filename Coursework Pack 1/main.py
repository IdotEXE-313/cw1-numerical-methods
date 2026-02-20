"""
This script is used to test your functions from CW1_solvers.
You are welcome to modify this script and add additional tests.
DO NOT SUBMIT THIS SCRIPT.
"""

import numpy as np
import matplotlib.pyplot as plt
import CW1_solvers as cw


#%% Question 1a

import numpy as np
import matplotlib as plt
import CW1_solvers as cw

print("Q1a TEST0 check your docstrings are completed:")
help(cw.fixedpoint_with_stopping)

#Question 1c
import numpy as np
import matplotlib as plt
import CW1_solvers as cw
g = lambda x: 1/3*(x**2-1)
p0 = 0.15
Nmax = 20
TOL = 1e-11
p_array = cw.fixedpoint_with_stopping(g,p0,Nmax,TOL)
print(f"Solution: {p_array}")


# Initialise
import numpy as np
import matplotlib.pyplot as plt
import CW1_solvers as cw
g = lambda x: 1/3*(x**2 - 1)
p0=1.0
Nmax=10
TOL=1e-11
print("Q1a TEST1 output without figure:")
p_array =cw.fixedpoint_with_stopping(g, p0, Nmax, TOL)
print(f"p_array:\n {p_array}")
print(f"p_array shape: {p_array.shape}\n")

Nmax=50
print("Q1a TEST2 output without figure (stop early):")
p_array =cw.fixedpoint_with_stopping(g, p0, Nmax, TOL)
print(f"p_array:\n {p_array}")
print(f"p_array shape: {p_array.shape}\n")


p=(3-np.sqrt(13))/2  # exact fixed point
#Q1a TEST3 produce error plot:
_, fig, ax=cw.fixedpoint_with_stopping(g, p0, Nmax, TOL, p)


#Error bound constants:
C=2
k=2/3
#Q1a TEST4 produce error and error bound plots:
_, fig, ax=cw.fixedpoint_with_stopping(g, p0, Nmax, TOL, p, C, k)



#%% Question 1b

#Q1b TEST. Checks how your answer to Question 1b is displayed:
print(cw.show_answer(cw.q1B_answer, 60))



#%% Question 1c

#Q1c TEST. Checks how your answer to Question 1c is displayed:
print(cw.show_answer(cw.q1C_answer, 60))



#%% Question 2a

# Initialise
f = lambda x: np.cos(x) - x
df = lambda x: -np.sin(x) - 1
p0 = 0
Nmax = 3
TOL =1e-16
print(f"Q2a TEST 1 output without figure")
p_array = cw.newton_with_stopping(f,df,p0,Nmax,TOL)
print(f"p_array:\n {p_array}")
print(f"p_array shape: {p_array.shape}\n")


Nmax = 10
TOL =1e-16
print(f"Q2a TEST 2 output without figure (stop early)")
p_array = cw.newton_with_stopping(f,df,p0,Nmax,TOL)
print(f"p_array:\n {p_array}")
print(f"p_array shape: {p_array.shape}")


#Exact solution:
p = np.float64(0.73908513321516064165531207047)
#Q2a TEST 3 output with plot:
p_array,fig,ax = cw.newton_with_stopping(f,df,p0,Nmax,TOL,p)

#Q2b TEST. Checks how your answer to Question 2b is displayed:
print(cw.show_answer(cw.q2B_answer, 60))



#%% Question 3


import numpy as np
import matplotlib.pyplot as plt
import CW1_solvers as cw

# Initialise
f = lambda x: x**2 - 2
p0 = 1
p1 = 2
Nmax = 12
TOL=1e-6
print(f"Q3 TEST 1")
p_array = cw.secant_with_stopping(f, p0, p1, Nmax, TOL)
print(f"p_array:\n {p_array}")
print(f"p_array shape: {p_array.shape}\n")

Nmax = 12
TOL=1e-16
print(f"Q3 TEST 2")
p_array = cw.secant_with_stopping(f, p0, p1, Nmax, TOL)
print(f"p_array:\n {p_array}")
print(f"p_array shape: {p_array.shape}")




#%% Question 4

# Initialise
p = np.float64(0.73908513321516064165531207047)
f = lambda x: x - np.cos(x)
df = lambda x: 1 + np.sin(x)
p0_sec=0.0
p1_sec=2.0
p0_newton=0.0
Nmax = 20
TOL=10**-16
#Q4 TEST 1 (plot)
fig, ax=cw.plot_convergence(p, f, df,  p0_newton, p0_sec, p1_sec,Nmax, TOL)



#Q4b TEST. Checks how your answer to Question 4b is displayed:
p = np.float64(-1.7692923542386314)
f = lambda x: x**3 - 2*x + 2
df = lambda x: 3*x**2 - 2
p0_sec = -2
p0_sec = -1
p0_newton = 0
Nmax = 30
TOL = 1e-16
fig,ax = cw.plot_convergence(p,f,df,p0_newton, p0_sec,p1_sec,Nmax,TOL)
print(cw.show_answer(cw.q4B_answer, 80))



#%% Question 5

import numpy as np
import CW1_solvers as cw

#Initialise
A = np.array([[1,-5,1],[10,0.0,20],[5,10,-1]], dtype=float)
b = np.array([[7],[6],[4]], dtype=float)
m=1
print("Q5a TEST 1")
tildeA, perm = cw.scaled_pivoting(A,b,m)
print(f"tildeA:\n {tildeA}\n")
print(f"perm:\n {perm}\n")

m=2
print("Q5a TEST 2")
tildeA, perm = cw.scaled_pivoting(A,b,m)
print(f"tildeA:\n {tildeA}\n")
print(f"perm:\n {perm}\n")


print("Q5b TEST 1")
x = cw.sp_solve(A, b)
print(f"x:\n {x}")

epsilon = 1e-16
A = np.array([[1,1,1],[1,1+epsilon,1],[1,1,1+epsilon]])
b = np.array([[1],[1+epsilon],[1+epsilon]])
x = cw.sp_solve(A,b)

print(f"x = \n{x}")

#Q5c Test. Checks how your answer to Question 5c is displayed:
print(cw.show_answer(cw.q5C_answer, 60))




# %%
