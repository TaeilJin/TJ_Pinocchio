import matplotlib.pyplot as plt
import numpy as np

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from scipy.spatial import ConvexHull, convex_hull_plot_2d

import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

import quadprog
#use array representation for binding eigen objects to python
import eigenpy
eigenpy.switchToNumpyArray()
from curves.plot import (plotBezier)
from curves.optimization import (constraint_flag, generate_integral_problem, integral_cost_flag, problem_definition,
                                setup_control_points)
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
            qp_C = -np.vstack([C, G]).T
            qp_b = -np.hstack([d, h])
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
def normalize(Ab):
    A = Ab[0]
    b = Ab[1]
    Ares = np.zeros(A.shape)
    bres = np.zeros(b.shape)
    for i in range(A.shape[0]):
        n = np.linalg.norm(A[i,:])
        if n <= 0.000001:
            n = 1.
        Ares[i,:] = A[i,:] / n; bres[i] = b[i] / n
    return Ares, bres
def normalizeMatAndVec(A,b):
    Ares = np.zeros(A.shape)
    bres = np.zeros(b.shape)
    for i in range(A.shape[0]):
        n = np.linalg.norm(A[i,:])
        if n <= 0.000001:
            n = 1.
        Ares[i,:] = A[i,:] / n; bres[i] = b[i] / n
    return Ares, bres
def plotConvexHull(hull2,pts2,ax,color ='b-'):
    for s in hull2.simplices:
        s = np.append(s, s[0])  # Here we cycle back to the first coordinate
        ax.plot(pts2[s, 0], pts2[s, 1], pts2[s, 2], color)
def createPointOnConvex(alpha1,alpha2,list_pos):
    point = alpha1*list_pos[0] + alpha2*list_pos[1] + (1 - (alpha1+alpha2))*list_pos[3]
    return point
def makeTimeInterval(n_Chulls,n_Splits):
    f_T_intv = 1 / float(n_Chulls)
    arr_T_intv = np.zeros(n_Splits)
    for i in range(n_Splits):
        arr_T_intv[i] = f_T_intv*float(i+1)
    return arr_T_intv
# C_i * B_j * X <= c_i - C_i*b_j
def stackInEqConstOfChull(array_CB, vec_CB, constrainedCurve,convexhull_ineq):

    dim = len(convexhull_ineq[0]) - 1
    if(dim< 1e-3):
        print 'error convex hull input'
    array_C_cur = np.array(convexhull_ineq)[:,0:(dim)]
    vec_C_cur = -np.array(convexhull_ineq)[:,(dim)]
    
    for j in range(constrainedCurve.nbWaypoints):
            mat_Info_Point_i = constrainedCurve.waypointAtIndex(j)

            arr_C_temp,vec_C_temp = createInEqConst(mat_Info_Point_i,False,array_C_cur,vec_C_cur)
            array_CB = np.vstack([array_CB,arr_C_temp])
            vec_CB = np.hstack([vec_CB,vec_C_temp])
    return array_CB, vec_CB
    # C*(Bx+b) <= c
def createInEqConst(matInfo,b_slackornot,array_C,vec_c):
    
    arr_C_temp = matInfo.B()
    vec_C_temp = matInfo.c()
    if(b_slackornot == True):
        arr_C_temp.fill(0)
        vec_C_temp.fill(0)
    else:
        vec_C_temp = vec_c - array_C.dot(matInfo.c())   
        arr_C_temp = np.dot(array_C,matInfo.B())   
            
    return arr_C_temp,vec_C_temp
def stackInEqConstOfChull_realtive(array_CB, vec_CB, constrainedCurve,p_s,convexhull_ineq):

    dim = len(convexhull_ineq[0]) - 1
    if(dim< 1e-3):
        print 'error convex hull input'
    array_C_cur = np.array(convexhull_ineq)[:,0:(dim)]
    vec_C_cur = -np.array(convexhull_ineq)[:,(dim)]
    
    for j in range(constrainedCurve.nbWaypoints):
            mat_Info_Point_i = constrainedCurve.waypointAtIndex(j)

            arr_C_temp,vec_C_temp = createInEqConst(mat_Info_Point_i,False,array_C_cur,vec_C_cur)
            # vec_C_temp = vec_C_cur - array_C_cur.dot(mat_Info_Point_i.c()-p_s)   
            # arr_C_temp = np.dot(array_C_cur,mat_Info_Point_i.B())
                       
            array_CB = np.vstack([array_CB,arr_C_temp])
            vec_CB = np.hstack([vec_CB,vec_C_temp])
    return array_CB, vec_CB

def createBezierChull(p_s,p_g,list_Chull,list_Chull_dev):
    # problem setting
    dim = len(p_s) # dimension = 3, 3D
    refDegree = 3   # degree = 3, polynomial we have to get 4 control points (start,end,c1,c2)
    pD = problem_definition(dim)
    pD.degree = refDegree
    
    #cost_Ab.cost.A & cost_Ab.cost.b
    cost_Ab = generate_integral_problem(pD, integral_cost_flag.ACCELERATION)
    problem = setup_control_points(pD)
    variableBezier = problem.bezier()
    problemSize = problem.numVariables * dim # dimension * control points 

    # convex hulls
    n_Chulls = len(list_Chull)
    n_Splits = n_Chulls - 1
    
    # split the original curve
    arr_T_intv = makeTimeInterval(n_Chulls,n_Splits)
    piecewiseCurve = variableBezier.split(arr_T_intv) 
    print arr_T_intv
    
    # set Inequality Constraints
    array_CB = np.array([],dtype=float).reshape(0,problemSize)
    vec_CB = []
   
    # assign the convexhull constraints
    for i in range(n_Chulls):
        k = i
        constrainedCurve = piecewiseCurve.curve_at_index(k)
        array_CB, vec_CB = stackInEqConstOfChull(array_CB,vec_CB,constrainedCurve,list_Chull[i].equations)
    
    # assign the derivative cosntraints
    n_Chulls_dev = len(list_Chull_dev)
    dev_Bezier = variableBezier.compute_derivate(1)
    
    # array_CB, vec_CB = stackInEqConstOfChull_realtive(array_CB,vec_CB,variableBezier,p_s,list_Chull_dev[0].equations)


    #assign final matrix
    ineqMatrix = array_CB
    ineqVector = vec_CB

    # set equality constraints
    array_CE = np.array([],dtype=float).reshape(0,problemSize)
    vec_CE = []

    matInfo_start = variableBezier.waypointAtIndex(0)
    array_CE = np.vstack([array_CE,matInfo_start.B()])
    vec_temp = p_s - matInfo_start.c()
    vec_CE = np.hstack([vec_CE,vec_temp])

    matInfo_end = variableBezier.waypointAtIndex(variableBezier.nbWaypoints-1)
    array_CE = np.vstack([array_CE,matInfo_end.B()])
    vec_temp = p_g - matInfo_end.c()
    vec_CE = np.hstack([vec_CE,vec_temp])

    eqMatrix = array_CE
    eqVector = vec_CE



    #
    res = quadprog_solve_qp(cost_Ab.cost.A, cost_Ab.cost.b, C = eqMatrix, d = eqVector, G=ineqMatrix, h = ineqVector)
    fitBezier = variableBezier.evaluate(res.reshape((-1,1)))

    piecewiseFit = fitBezier.split(arr_T_intv)
    plotBezier(piecewiseFit.curve_at_index(0), ax = ax, linewidth=4., color = "b")
    plotBezier(piecewiseFit.curve_at_index(1), ax = ax, linewidth=4., color = "r")
    # plotBezier(piecewiseFit.curve_at_index(2), ax = ax, linewidth=4., color = "b")
    # plotBezier(piecewiseFit.curve_at_index(3), ax = ax, linewidth=4., color = "r")

def createBezierChullwithSlack(p_s,p_g,list_Chull,list_Chull_dev):
    # problem setting
    dim = len(p_s) # dimension = 3, 3D
    refDegree = 3   # degree = 3, polynomial we have to get 4 control points (start,end,c1,c2)
    pD = problem_definition(dim)
    pD.degree = refDegree
    pD.init_pos = p_s
    pD.end_pos = p_g
    # pD.flag = constraint_flag.INIT_POS | constraint_flag.END_POS #| constraint_flag.INIT_VEL #| constraint_flag.END_VEL

    #cost_Ab.cost.A & cost_Ab.cost.b
    cost_Ab = generate_integral_problem(pD, integral_cost_flag.ACCELERATION)
    problem = setup_control_points(pD)
    variableBezier = problem.bezier()
    problemSize = problem.numVariables * dim # dimension * control points 

    # #changing to solve slack variable
    num_slack = 1
    mat_origin = cost_Ab.cost.A 
    vec_origin = cost_Ab.cost.b 

    newCostP = np.zeros((mat_origin.shape[0] + num_slack, mat_origin.shape[1] + num_slack))
    newCostP[0:mat_origin.shape[0],:mat_origin.shape[1]] = mat_origin
    newCostP[newCostP.shape[0]-num_slack, newCostP.shape[1]-num_slack] = 1e-4

    newCostq = np.zeros(len(vec_origin)+num_slack)
    newCostq[:len(vec_origin)] = vec_origin
    newCostq[-1] = -1e+5
    
    # # convex hulls
    n_Chulls = len(list_Chull)
    n_Splits = n_Chulls - 1
    
    # split the original curve
    arr_T_intv = makeTimeInterval(n_Chulls,n_Splits)
    piecewiseCurve = variableBezier.split(arr_T_intv) 
    print arr_T_intv
    
    # set Inequality Constraints
    array_CB = np.array([],dtype=float).reshape(0,problemSize)
    vec_CB = []

    # print array_CB.shape
    # assign the convexhull constraints
    for i in range(n_Chulls):
        k = i
        constrainedCurve = piecewiseCurve.curve_at_index(k)
        array_CB, vec_CB = stackInEqConstOfChull(array_CB,vec_CB,constrainedCurve,list_Chull[i].equations)
    array_CB, vec_CB = normalizeMatAndVec(array_CB,vec_CB)    
   
    # #slack vector
    # vec_slack = np.zeros((array_CB.shape[0],1))
    # vec_slack.fill(0)
    # vec_slack[36:48] =1
    # vec_slack[48:60] =1
    # array_CB = np.hstack([array_CB,vec_slack])

    # vec_slack2 = np.zeros((array_CB.shape[1]))
    # vec_slack2[-1] = -1
    # array_CB = np.vstack([array_CB,vec_slack2])

    # new_vec_CB = np.zeros((array_CB.shape[0]))
    # new_vec_CB[:len(vec_CB)] = vec_CB
    # new_vec_CB[-1] = 0
    # vec_CB = new_vec_CB
    
    # print array_CB.shape
    # print vec_CB.shape

    #assign final matrix
    ineqMatrix = array_CB
    ineqVector = vec_CB

    # # set equality constraints
    array_CE = np.array([],dtype=float).reshape(0,problemSize)
    vec_CE = []

    matInfo_start = variableBezier.waypointAtIndex(0)
    vec_norm = np.zeros((3,1))
    vec_norm[2] = 1 
    
    vec_norm = np.transpose(vec_norm)
    print vec_norm.shape
    print matInfo_start.B().shape
    out3 = vec_norm.dot(matInfo_start.B())
    print out3
    array_CE = np.vstack([array_CE,out3])

    vec_temp = -1*vec_norm.dot(matInfo_start.c())
    vec_CE = np.hstack([vec_CE,vec_temp])

    # matInfo_end = variableBezier.waypointAtIndex(variableBezier.nbWaypoints-1)
    # array_CE = np.vstack([array_CE,matInfo_end.B()])
    # vec_temp = p_g - matInfo_end.c()
    # vec_CE = np.hstack([vec_CE,vec_temp])
    
    # array_CE, vec_CE = normalizeMatAndVec(array_CE,vec_CE)

    # print array_CE.shape
    # print vec_CE.shape
    # #
    # # constrainedCurve = piecewiseCurve.curve_at_index(0)
    # # array_C_cur = np.array(list_Chull[0].equations)[:,0:(dim)]
    # # vec_C_cur = -np.array(list_Chull[0].equations)[:,(dim)]
    
    # # mat_Info_Point_i = constrainedCurve.waypointAtIndex(2)
    # # arr_C_temp,vec_C_temp = createInEqConst(mat_Info_Point_i,False,array_C_cur,vec_C_cur)
    # # array_CE = np.vstack([array_CE,arr_C_temp])
    # # vec_CE = np.hstack([vec_CE,vec_C_temp])

    # # constrainedCurve = piecewiseCurve.curve_at_index(1)
    # # array_C_cur = np.array(list_Chull[1].equations)[:,0:(dim)]
    # # vec_C_cur = -np.array(list_Chull[1].equations)[:,(dim)]
    
    # # mat_Info_Point_i = constrainedCurve.waypointAtIndex(0)
    # # arr_C_temp,vec_C_temp = createInEqConst(mat_Info_Point_i,False,array_C_cur,vec_C_cur)
    # # array_CE = np.vstack([array_CE,arr_C_temp])
    # # vec_CE = np.hstack([vec_CE,vec_C_temp])

    # print array_CE.shape
    # print vec_CE.shape
    # # array_CE, vec_CE = normalizeMatAndVec(array_CE,vec_CE)

    # vec_slack = np.zeros((array_CE.shape[0],1))
    # vec_slack.fill(0)
    # for i in range(6):
    #     vec_slack[i]=0

    # array_CE = np.hstack([array_CE,vec_slack])

    
    eqMatrix = array_CE
    eqVector = vec_CE

    array_C_cur = np.array(list_Chull[0].equations)[:,0:(dim)]
    vec_C_cur = -np.array(list_Chull[0].equations)[:,(dim)]
    print array_C_cur
    print vec_C_cur
    # #
    res = quadprog_solve_qp(mat_origin , vec_origin, G=ineqMatrix, h = ineqVector, C=eqMatrix,d = eqVector)

    fitBezier = variableBezier.evaluate(res.reshape((-1,1)))

    piecewiseFit = fitBezier.split(arr_T_intv)
    plotBezier(piecewiseFit.curve_at_index(0), ax = ax, linewidth=4., color = "b",showControlPoints=True)
    plotBezier(piecewiseFit.curve_at_index(1), ax = ax, linewidth=4., color = "r",showControlPoints=True)
    # # plotBezier(piecewiseFit.curve_at_index(2), ax = ax, linewidth=4., color = "b")
    # # plotBezier(piecewiseFit.curve_at_index(3), ax = ax, linewidth=4., color = "r")

    
# prepare some coordinates
x, y, z = np.indices((8, 8, 8))

# draw cuboids in the top left and bottom right corners, and a link between them
cube1 = (x < 8) & (y < 2) & (z < 1)
cube2 = (x < 8) & (2<=y) & (y<4) & (z <2)
cube3 = (x < 8) & (4<=y) & (y<7) & (z< 4)
cube4 = (x < 8) & (6<=y) & (y<=8) & (z< 6)

# combine the objects into a single boolean array
voxels = cube1 | cube2 | cube3 |cube4
# set the colors of each object
colors = np.empty(voxels.shape, dtype=object)

colors[cube1] = 'blue'
colors[cube2] = 'green'
colors[cube3] = 'red'
colors[cube4] = 'orange'

# 8 points defining the cube corners
pts = np.array([[0, 0, 1], [8, 0, 1], [0, 2, 1], [8, 2, 1],
                [0, 0, 8], [8, 0, 8], [0, 2, 8], [8, 2, 8], ])
hull = ConvexHull(pts)

pts2 = np.array([[0, 1.5, 2], [8, 1.5, 2], [0, 4, 2],  [8, 4, 2],
                [0, 1.5, 8], [0, 4, 8], [8, 1.5, 8], [8, 4, 8], ])
hull2 = ConvexHull(pts2)

pts3 = np.array([[0, 3.5, 4], [0, 6, 4], [8, 3.5, 4], [8, 6, 4],
                [0, 3.5, 8], [0, 6, 8], [8, 3.5, 8], [8, 6, 8], ])
hull3 = ConvexHull(pts3)

pts4 = np.array([[0, 5.5, 6], [8, 5.5, 6],[0, 8, 6], [8, 8, 6],
                [0, 5.5, 8], [0, 8, 8], [8, 5.5, 8], [8, 8, 8], ])
hull4 = ConvexHull(pts4)
# 8 points defining the derivative convex hull
pts_dev_s = np.array([[0, 0, 0], [1e-3, 0, 0], [0, 1, 0], [1e-3, 1, 0],
                [0, 0, 8], [1e-3, 0, 8], [0, 1, 8], [1e-3, 1, 8], ])
hull_dev_s = ConvexHull(pts_dev_s)
# kinematics constraints
pts_dev_k = np.array([[-100, 0, 0], [100, 100, 0], [-100, 108, 0], [100, 108, 0],
                [-100, 100, 100], [100, 100, 100], [-100, 108, 100], [100, 108, 100], ])
hull_dev_k = ConvexHull(pts_dev_k)

in_list_Chull=[]
in_list_Chull.append(hull)
in_list_Chull.append(hull2)
# in_list_Chull.append(hull3)
# in_list_Chull.append(hull4)

in_list_Chull_dev=[]
in_list_Chull_dev.append(hull_dev_k)
# and plot everything
fig = plt.figure()
ax = fig.gca(projection='3d')

# 12 = 2 * 6 faces are the simplices (2 simplices per square face)
plotConvexHull(hull,pts,ax,'r-')
plotConvexHull(hull2,pts2,ax,'g-')
plotConvexHull(hull3,pts3,ax,'b-')
plotConvexHull(hull4,pts4,ax,'k-')
# ax.voxels(voxels, facecolors=colors, edgecolor='k')

# make points
pt_start = createPointOnConvex(0.5,0.4,pts)
pt_end = createPointOnConvex(0.5,0.1,pts2)
ax.scatter(pt_start[0],pt_start[1],pt_start[2])
ax.scatter(pt_end[0],pt_end[1],pt_end[2])
createBezierChull(pt_start,pt_end,in_list_Chull,in_list_Chull_dev)
ax.plot([pt_start[0], pt_end[0]], [pt_start[1],pt_end[1]],zs=[pt_start[2],pt_end[2]],color ='orange',linestyle='dashed',linewidth =2)

plt.draw()

# Make axis label
for i in ["x", "y", "z"]:
    eval("ax.set_{:s}label('{:s}')".format(i, i))

plt.show()