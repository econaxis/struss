# Bridge Simulator

## Context

My university physics class required me to design the cheapest bridge that could support some weight over some distance. With my extensive experience in designing bridges in *Poly Bridge 2*, I thought it'd be easy. I immediately drew up designs.

To analyze forces on bridges, we'd learned about static structural analysis using the method of joints. Assuming all joints are in static equilibrium it is easy to solve the forces on joints as a linear system. Doing this by hand, however, sucked all the fun out of making bridges ~~watching them break~~. 

Therefore, I wanted a computer program to solve the forces on all joints of any bridge design quickly. It would read a bridge from a 2D CAD drawing of a bridge (DXF format), then print out all the forces.

## How?

At every joint identified in the DXF drawing, I would generate linear equations. Then, I used the `sympy` package to symbolically solve all linear equations. However, the problems in this approach were:
 - symbolic solving is too slow
 - sympy handles underdetermined or overdetermined systems very poorly

I resorted to a more manual approach by constructing a large matrix from the generated linear equations, and used the `numpy` library to least-squares `rref` the matrix. This worked much better and faster than sympy.

## Being Lazy

I soon realized I could use this program to simulate all possible variations of bridges, and select the lowest-cost one. I did this by jittering a random joint in a random direction. If the cost of the new bridge is lower than that of the old one, the new one is set as the old one. The procedure is then repeated until convergence.

If I were feeling smart, I would call this stochastic gradient descent. But actually, it's just tumbling down N-dimensional hyperspace until we can't tumble down anymore. 


