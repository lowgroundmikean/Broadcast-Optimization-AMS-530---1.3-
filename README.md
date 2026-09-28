AMS Project 1 - Problem 1.3

Optimal broadcast of 16 double-precision values on the optimal (N = 32, k = 3) regular graph of Zhang, Xu and Deng, implemented in MPI.
Result: the broadcast completes in S = 6 rounds, activating 1, 2, 4, 6, 10, 8 edges per round for a total of 31. 
Six rounds is optimal: a Fibonacci-type counting bound shows that no cubic graph on 32 vertices can be broadcast in fewer, 
and the schedule attains the bound at every intermediate round, not only at termination.
