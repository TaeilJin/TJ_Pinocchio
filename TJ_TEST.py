#---------------
#-1.-----curve library setting
# importing classical numpy objects
from numpy import zeros, array, identity, dot
from numpy.linalg import norm
import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})

#use array representation for binding eigen objects to python
import eigenpy
eigenpy.switchToNumpyArray()

#importing the bezier curve class
from curves import (bezier)

#importing tools to plot bezier curves
from curves.plot import (plotBezier)
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

#We describe a degree 3 curve as a Bezier curve with 4 control points
waypoints = array([[1., 2., 3.], [-4., -5., -6.], [4., 5., 6.], [7., 8., 9.]]).transpose()
ref = bezier(waypoints)

#plotting sampled points on the curve
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

#plotting the curve with its control points
plotBezier(ref,showControlPoints = True, color="g",ax=ax)

#plotting sampled points
numSamples = 10; fNumSamples = float(numSamples)
ptsTime = [ (ref(float(t) / fNumSamples), float(t) / fNumSamples) for t in range(numSamples+1)]
print ptsTime
points_sampled = np.asarray(ptsTime)[:,0]
x = np.array([wp[0] for wp in points_sampled])
y = np.array([wp[1] for wp in points_sampled])
z = np.array([wp[2] for wp in points_sampled])
ax.scatter(x, y, z, color="r")



#---------------
#-2.-----optimizing the beizer curve following sampled points
from curves.optimization import (problem_definition, setup_control_points)

#dimension of our problem (here 3 as our curve is 3D)
dim = 3 # dimension = 3, 3D
refDegree = 3   # degree = 3, polynomial we have to get 4 control points (start,end,c1,c2)
pD = problem_definition(dim)
pD.degree = refDegree #we want to fit a curve of the same degree as the reference curve for the sanity check

#generates the variable bezier curve with the parameters of problemDefinition
problem = setup_control_points(pD)
#for now we only care about the curve itself
variableBezier = problem.bezier()

#least square form of ||Ax-b||**2 
def to_least_square(A, b):
    return dot(A.T, A), - dot(A.T, b)

def genCost(variableBezier, ptsTime):
    #first evaluate variableBezier for each time sampled
    allsEvals = [(variableBezier(time), pt) for (pt,time) in ptsTime]
    #then compute the least square form of the cost for each points
    allLeastSquares = [to_least_square(el.B(), el.c() + pt) for (el, pt) in  allsEvals] # el.c() + pt ? 
    #and finally sum the costs
    Ab = [sum(x) for x in zip(*allLeastSquares)]
    return Ab[0], Ab[1]
A, b = genCost(variableBezier, ptsTime)

import quadprog
from numpy import array, hstack, vstack

def quadprog_solve_qp(P, q, G=None, h=None, C=None, d=None, verbose=False):
    """
    min (1/2)x' P x + q' x
    subject to  G x <= h
    subject to  C x  = d
    """
    # qp_G = .5 * (P + P.T)   # make sure P is symmetric
    qp_G = .5 * (P + P.T)  # make sure P is symmetric
    qp_a = -q
    qp_C = None
    qp_b = None
    meq = 0
    if C is not None:
        if G is not None:
            qp_C = -vstack([C, G]).T
            qp_b = -hstack([d, h])
        else:
            qp_C = -C.transpose()
            qp_b = -d
        meq = C.shape[0]
    elif G is not None:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
    res = quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)
    if verbose:
        return res
    # print('qp status ', res)
    return res[0]

res = quadprog_solve_qp(A, b)

def evalAndPlot(variableBezier, res):
    fitBezier = variableBezier.evaluate(res.reshape((-1,1)) ) # creating beizer variables (maybe control points) using res class
    ##plot reference curve in blue, fitted curve in green
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection="3d")  
    plotBezier(ref, ax = ax, linewidth=4.) #thicker line to visualize overlap
    plotBezier(fitBezier, ax = ax, color ="g", linewidth=3.) 
    #plt.show()
    return fitBezier
    
fitBezier = evalAndPlot(variableBezier, res)

#---------------
#-3.-----optimizing the beizer curve following sampled points in lesser degree of problem
pD.degree = refDegree - 1 # quadratic form
problem = setup_control_points(pD)
variableBezier = problem.bezier()

A, b = genCost(variableBezier,ptsTime)
res = quadprog_solve_qp(A,b)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")  
##fitBezier = evalAndPlot(variableBezier,res)

#-3.1 --- start end position constraint
from curves.optimization import constraint_flag

pD.flag = constraint_flag.INIT_POS | constraint_flag.END_POS
#set initial position
pD.init_pos = array([ptsTime[ 0][0]]).T
#set end position
pD.end_pos   = array([ptsTime[-1][0]]).T # -1 refers to the last index, -2 refers to the second last index and so on
problem = setup_control_points(pD)
variableBezier = problem.bezier()
A, b = genCost(variableBezier,ptsTime)
res = quadprog_solve_qp(A,b)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d") 

##fitBezier = evalAndPlot(variableBezier,res)

#-3.2.1 --- puts other constraints ( we should increase the degree )
#values are 0 by default, so if the constraint is zero this can be skipped
pD.init_vel = array([[0., 0., 0.]]).T
pD.init_acc = array([[0., 0., 0.]]).T
pD.end_vel = array([[0., 0., 0.]]).T
pD.end_acc = array([[0., 0., 0.]]).T
pD.flag = constraint_flag.END_POS | constraint_flag.INIT_POS | constraint_flag.INIT_VEL  | constraint_flag.END_VEL  | constraint_flag.INIT_ACC   | constraint_flag.END_ACC

pD.degree = refDegree + 3
prob = setup_control_points(pD) # why degree of problem is same with number of constraints?
variableBezier = prob.bezier()
A, b = genCost(variableBezier,ptsTime)
res = quadprog_solve_qp(A,b)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d") 

#fitBezier = evalAndPlot(variableBezier,res)

#-3.2.2 --- we need more control points to guarantee the constraints
pD.degree = refDegree + 60
prob = setup_control_points(pD)
variableBezier = prob.bezier()
A, b = genCost(variableBezier, ptsTime)
#regularization matrix 
reg = identity(A.shape[1]) * 0.001
res = quadprog_solve_qp(A+reg, b) # without regularization?
##fitBezier = evalAndPlot(variableBezier, res)

#-3.2.3 --- puts inequality constraints 
#set initial / terminal constraints
pD.flag = constraint_flag.END_POS | constraint_flag.INIT_POS
pD.degree = refDegree
prob = setup_control_points(pD)
variableBezier = prob.bezier()

#get value of the curve first order derivative at t = 0.8
t08Constraint = variableBezier.derivate(0.8,1)
target = zeros(3) 

A, b = genCost(variableBezier, ptsTime)
#solve optimization problem with quadprog

res = quadprog_solve_qp(A, b, C=t08Constraint.B(), d=target - t08Constraint.c()) # why d=target - t08Constraint.c() ?

fitBezier = evalAndPlot(variableBezier, res)

assert norm(fitBezier.derivate(0.8,1) - target) <= 0.001


#-4.1.1 --- splitting the reference beizer curve
# #returns a curve composed of the split curves, 2 in our case
piecewiseCurve = ref.split(array([[0.6]]).T)

#displaying the obtained curves

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")  

#first, plotting the complete piecewiseCurve is equivalent
plotBezier(piecewiseCurve, ax = ax, linewidth=10., color = "b")
plotBezier(piecewiseCurve.curve_at_index(0), ax = ax, linewidth=4., color = "r")
plotBezier(piecewiseCurve.curve_at_index(1), ax = ax, linewidth=4., color = "orange")

#-4.1.2 --- optimized bezier curve 
#first, split the variable curve
piecewiseCurve = variableBezier.split(array([[0.4, 0.8]]).T)

constrainedCurve = piecewiseCurve.curve_at_index(1)

#find the number of variables
problemSize = prob.numVariables * dim # what is this? dimension * degree ?
print problemSize
#find the number of constraints, as many as waypoints
nConstraints = constrainedCurve.nbWaypoints


waypoints = constrainedCurve.waypoints() # what kind of variables inside this class?

ineqMatrix = zeros((nConstraints, problemSize))
ineqVector = zeros(nConstraints)

print ' ineq may be (3,4)'
print ineqMatrix.shape


#finding the z equation of each control point
for i in range(nConstraints):
    wayPoint = constrainedCurve.waypointAtIndex(i)
    ineqMatrix[i,:] = wayPoint.B()[2,:]
    ineqVector[i] =  -wayPoint.c()[2] # what is the B,c of way point? what is the geometrical meaning?
    
 
res = quadprog_solve_qp(A, b, G=ineqMatrix, h = ineqVector)
fitBezier = variableBezier.evaluate(res.reshape((-1,1)) ) 


#now plotting the obtained curve, in red the concerned part
piecewiseFit = fitBezier.split(array([[0.4, 0.8]]).T)
plotBezier(piecewiseFit.curve_at_index(0), ax = ax, linewidth=4., color = "b")
plotBezier(piecewiseFit.curve_at_index(1), ax = ax, linewidth=4., color = "r")
plotBezier(piecewiseFit.curve_at_index(2), ax = ax, linewidth=4., color = "b")

#plotting the plane z = 0
xx, yy = np.meshgrid(range(20), range(20))

# calculate corresponding z
z = (0 * xx - 0 * yy )

# plot the surface
# ax.plot_surface(xx, yy, z, alpha=0.2)


plt.show()