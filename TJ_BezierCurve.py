from matplotlib import pyplot as plt
from numpy import zeros, array, matrix, identity, dot
from numpy.linalg import norm
import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})

#use array representation for binding eigen objects to python
import eigenpy
eigenpy.switchToNumpyArray()

#importing the bezier curve class
from curves import (bezier)
from curves.plot import (plotBezier2D)
# creating segmented bezier curve
from curves.optimization import (constraint_flag, generate_integral_problem, integral_cost_flag, problem_definition,
                                    setup_control_points)
import quadprog
from numpy import array, hstack, vstack

def plotLine(in_LB):
    plt.plot(in_LB.point_x,in_LB.point_y,'ro')
    plt.plot(in_LB.point_x,in_LB.point_y,linewidth =2.0,color ='orange')
    plt.draw()

def MakeSpline2D(in_LB):
    print('draw spline ')
    
    # #We describe a degree 3 curve as a Bezier curve with 4 control points
    distance = in_LB.point_x[-1] - in_LB.point_x[0]
    distance_x = abs(distance)
    distance = in_LB.point_y[-1] - in_LB.point_y[0]
    distance_y = abs(distance)

    waypoints = array([[in_LB.point_x[0],in_LB.point_y[0],0],
        [in_LB.point_x[0] + distance_x,in_LB.point_y[0],0],
        [in_LB.point_x[0],in_LB.point_y[0]+distance_y,0], 
        [in_LB.point_x[-1],in_LB.point_y[-1],0]]).transpose()
    ref = bezier(waypoints)

    plotBezier2D(ref,ax=ax_mouse,showControlPoints=True)

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

def MakeBeizerConvexhull2D(in_LB):
    #dimension of our problem (here 3 as our curve is 3D)
    dim = 2 # dimension = 3, 3D
    refDegree = 3   # degree = 3, polynomial we have to get 4 control points (start,end,c1,c2)
    pD = problem_definition(dim)
    pD.degree = refDegree #we want to fit a curve of the same degree as the reference curve for the sanity check
    pD.flag = constraint_flag.INIT_POS | constraint_flag.END_POS
    #set initial position
    pD.init_pos = array([in_LB.point_x[0],in_LB.point_y[0]])
    #set end position
    pD.end_pos   = array([in_LB.point_x[-1],in_LB.point_y[-1]]) # -1 refers to the last index, -2 refers to the second last index and so on
    problem = setup_control_points(pD)
    #generates the variable bezier curve with the parameters of problemDefinition
    variableBezier = problem.bezier()

    #We describe a degree 3 curve as a Bezier curve with 4 control points
    distance = in_LB.point_x[-1] - in_LB.point_x[0]
    distance_x = abs(distance)
    distance = in_LB.point_y[-1] - in_LB.point_y[0]
    distance_y = abs(distance)

    waypoints = array([[in_LB.point_x[0],in_LB.point_y[0]],
        [in_LB.point_x[0] + distance_x,in_LB.point_y[0]],
        [in_LB.point_x[0],in_LB.point_y[0]+distance_y],
        [in_LB.point_x[-1],in_LB.point_y[-1]]]).transpose() #[in_LB.point_x[0],in_LB.point_y[0]+distance_y], 
    ref = bezier(waypoints)
    #plotting sampled points
    numSamples = 10; fNumSamples = float(numSamples)
    ptsTime = [ (ref(float(t) / fNumSamples), float(t) / fNumSamples) for t in range(numSamples+1)]
    
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
    costAb = generate_integral_problem(pD, integral_cost_flag.ACCELERATION)
    
    
    problemSize = problem.numVariables * dim # what is this? dimension * degree ?
    
    convexhull_1_ineq = array(in_LB.convexhull1.equations)
    convexhull_2_ineq = array(in_LB.convexhull2.equations)

    nConvexEq = convexhull_1_ineq.shape[0]

    ineqMatrix = zeros((nConvexEq*4 + nConvexEq * 4 * 2 + nConvexEq*4, problemSize)) # C1,C2[4*2*2]
    ineqVector = zeros(nConvexEq*4 + nConvexEq * 4 * 2 + nConvexEq*4)

    # ineqMatrix = zeros((convexhull_1_ineq.shape[0] * 4 * 2, problemSize)) # C1,C2[4*2*2]
    # ineqVector = zeros(convexhull_1_ineq.shape[0] * 4 * 2)
    
    
    array_CB = np.array([],dtype=float).reshape(0,4)

    vec_CB = []

    piecewiseCurve = variableBezier.split(array([[0.3,0.6]]).T) 
    # left side
    constrainedCurve = piecewiseCurve.curve_at_index(0)
    array_C_cur = array(convexhull_1_ineq)[:,0:2]
    vec_C_cur = -array(convexhull_1_ineq)[:,2]
    for k in range(4):
            mat_Info_Point_i = constrainedCurve.waypointAtIndex(k)

            vec_C_temp = vec_C_cur - array_C_cur.dot(mat_Info_Point_i.c())   
            arr_C_temp = dot(array_C_cur,mat_Info_Point_i.B())
                       
            array_CB = np.vstack([array_CB,arr_C_temp])
            vec_CB = np.hstack([vec_CB,vec_C_temp])

    # overlapped space
    constrainedCurve = piecewiseCurve.curve_at_index(1)
    for i in range(2):
        if i == 0 :
            array_C_cur = array(convexhull_1_ineq)[:,0:2]
            vec_C_cur = -array(convexhull_1_ineq)[:,2]
        else :
            array_C_cur = array(convexhull_2_ineq)[:,0:2]
            vec_C_cur = -array(convexhull_2_ineq)[:,2]
        for k in range(4):
            print array_C_cur
            mat_Info_Point_i = constrainedCurve.waypointAtIndex(k)

            vec_C_temp = vec_C_cur - array_C_cur.dot(mat_Info_Point_i.c())   
            arr_C_temp = dot(array_C_cur,mat_Info_Point_i.B())
                       
            array_CB = np.vstack([array_CB,arr_C_temp])
            vec_CB = np.hstack([vec_CB,vec_C_temp])
    # right side
    constrainedCurve = piecewiseCurve.curve_at_index(2)
    array_C_cur = array(convexhull_2_ineq)[:,0:2]
    vec_C_cur = -array(convexhull_2_ineq)[:,2]
   
    for k in range(4):
            mat_Info_Point_i = constrainedCurve.waypointAtIndex(k)

            vec_C_temp = vec_C_cur - array_C_cur.dot(mat_Info_Point_i.c())   
            arr_C_temp = dot(array_C_cur,mat_Info_Point_i.B())
                       
            array_CB = np.vstack([array_CB,arr_C_temp])
            vec_CB = np.hstack([vec_CB,vec_C_temp])
    
    ineqMatrix = array_CB
    ineqVector = vec_CB
    
    res = quadprog_solve_qp(costAb.cost.A, costAb.cost.b, G=ineqMatrix, h = ineqVector)
    fitBezier = variableBezier.evaluate(res.reshape((-1,1)) ) 


    #now plotting the obtained curve, in red the concerned part
    piecewiseFit = fitBezier.split(array([[0.3, 0.6]]).T)
    plotBezier2D(piecewiseFit.curve_at_index(0), ax = ax_mouse,  color = "b")
    plotBezier2D(piecewiseFit.curve_at_index(1), ax = ax_mouse,  color = "r")
    plotBezier2D(piecewiseFit.curve_at_index(2), ax = ax_mouse,  color = "b")
    plt.draw()

def makeTimeInterval(n_Chulls,n_Splits):
    f_T_intv = 1 / float(n_Chulls)
    arr_T_intv = zeros(n_Splits)
    for i in range(n_Splits):
        arr_T_intv[i] = f_T_intv*float(i+1)
    return arr_T_intv

# C_i * B_j * X <= c_i - C_i*b_j
def stackInEqConstOfChull(array_CB, vec_CB, constrainedCurve,convexhull_ineq):
    array_C_cur = array(convexhull_ineq)[:,0:2]
    vec_C_cur = -array(convexhull_ineq)[:,2]
    
    for j in range(constrainedCurve.nbWaypoints):
            mat_Info_Point_i = constrainedCurve.waypointAtIndex(j)

            vec_C_temp = vec_C_cur - array_C_cur.dot(mat_Info_Point_i.c())   
            arr_C_temp = dot(array_C_cur,mat_Info_Point_i.B())
                       
            array_CB = np.vstack([array_CB,arr_C_temp])
            vec_CB = np.hstack([vec_CB,vec_C_temp])
    return array_CB, vec_CB

def CreateBeizierChull(p_s,p_g,list_Chull):
    #problem setting
    dim = 2 # dimension = 3, 3D
    refDegree = 3   # degree = 3, polynomial we have to get 4 control points (start,end,c1,c2)
    pD = problem_definition(dim)
    pD.degree = refDegree
    
    pD.init_pos = p_s
    pD.end_pos = p_g
    pD.flag = constraint_flag.INIT_POS | constraint_flag.END_POS #| constraint_flag.INIT_VEL #| constraint_flag.END_VEL

    # cost_Ab.cost.A & cost_Ab.cost.b
    cost_Ab = generate_integral_problem(pD, integral_cost_flag.ACCELERATION)
    problem = setup_control_points(pD)
    variableBezier = problem.bezier()
    problemSize = problem.numVariables * dim # dimension * control points 
   
    # convex hulls
    n_Chulls = len(list_Chull)
    n_Splits = n_Chulls - 1
    nConvexEq = list_Chull[0].equations.shape[0]
    print nConvexEq
    
    # split the original curve
    arr_T_intv = makeTimeInterval(n_Chulls,n_Splits)
    piecewiseCurve = variableBezier.split(arr_T_intv) 
   
    # Equality Equation Matrix
    # n_eq_rows = n_Chulls*nConvexEq*(refDegree+1) + refDegree*dim
    # n_eq_cols = problemSize
    # ineqMatrix = zeros((n_eq_rows,n_eq_cols))
    # ineqVector = zeros(n_eq_rows)

    array_CB = np.array([],dtype=float).reshape(0,problemSize)
    vec_CB = []
   
    # assign the convexhull constraints
    for i in range(n_Chulls):
        k = i
        constrainedCurve = piecewiseCurve.curve_at_index(k)
        array_CB, vec_CB = stackInEqConstOfChull(array_CB,vec_CB,constrainedCurve,list_Chull[i].equations)
   
    # assign the derivative cosntraints
    dev_Bezier = variableBezier.compute_derivate(1)
    print 'B_t'
    print variableBezier.nbWaypoints
    print 'B_dot_t'
    print dev_Bezier.nbWaypoints
    
    vecDesC = zeros(2)
    des_dev_x = 100.0
    des_dev_y = 100.0
    for i in range(refDegree):
        matBc = dev_Bezier.waypointAtIndex(i)
        
        if(i == 0):
            des_dev_x = 4.0
        # if(i == 1):
        #     des_dev_x = -0.1
        if(i == 2):
            des_dev_x = 2.0

        array_CB = np.vstack([array_CB,matBc.B()])
        
        vecDesC[0] = des_dev_x - matBc.c()[0]
        vecDesC[1] = des_dev_y - matBc.c()[1]
        
        vec_CB = np.hstack([vec_CB,vecDesC])

    # assign final matrix
    ineqMatrix = array_CB
    ineqVector = vec_CB

    #
    res = quadprog_solve_qp(cost_Ab.cost.A, cost_Ab.cost.b, G=ineqMatrix, h = ineqVector)
    fitBezier = variableBezier.evaluate(res.reshape((-1,1)))
    
    #now plotting the obtained curve, in red the concerned part
    piecewiseFit = fitBezier.split(arr_T_intv)
    plotBezier2D(piecewiseFit.curve_at_index(0), ax = ax_mouse,  color = "b" ,showControlPoints=True)
    plotBezier2D(piecewiseFit.curve_at_index(1), ax = ax_mouse,  color = "r" ,showControlPoints=True)
    plotBezier2D(piecewiseFit.curve_at_index(2), ax = ax_mouse,  color = "b" ,showControlPoints=True)
    
    #
    # ax_dev = fig.
    dev_fitBezier = fitBezier.compute_derivate(1)
    plotBezier2D(dev_fitBezier, ax = ax_dev,  color = "g", showControlPoints=True )
    plt.show() 


class LineBuilder:
    def __init__(self, in_line, in_convexhull_1, in_convexhull_2,in_list_convexhull):
        self.line = in_line
        self.point_x = []
        self.point_y = []
        self.point_c_1 =[]
        self.point_c_2 = []
        self.nCnt = 0
        self.cid = in_line.figure.canvas.mpl_connect('button_press_event', self)
        self.convexhull1 = in_convexhull_1
        self.convexhull2 = in_convexhull_2
        self.list_convexhull = in_list_convexhull
    def __call__(self, event):
        print('click', event)
        self.nCnt+=1

        if event.inaxes!=self.line.axes: return
        self.point_x.append(event.xdata)
        self.point_y.append(event.ydata)
        self.line.set_data(self.point_x, self.point_y)
        if self.nCnt == 2 :
            plotLine(linebuilder)
            
            CreateBeizierChull(array([self.point_x[0],self.point_y[0]]),array([self.point_x[1],self.point_y[1]]),self.list_convexhull)
            self.point_x[:] = []
            self.point_y[:] = []
            self.nCnt =0
         


fig = plt.figure(1)
ax_mouse = fig.add_subplot(111)
ax_mouse.set_xlim(0,2) 
ax_mouse.set_ylim(0,2)
ax_mouse.grid(color='gray', linestyle='--', linewidth=2)
ax_mouse.set_title('click to build line segments')

line, = ax_mouse.plot([0], [0])  # empty line


from matplotlib.path import Path
import matplotlib.patches as patches

verts1 = [
   (0.5, 0.0),  # left, bottom
   (0.5, 0.5),  # left, top
   (2.0, 0.5),  # right, top
   (2.0, 0.0),  # right, bottom
   (0.5, 0.0),  # ignored
]
verts2 = [
   (1.0, 0.5),  # left, bottom
   (1.00, 1.25),  # left, top
   (2.0, 1.25),  # right, top
   (2.0, 0.5),  # right, bottom
   (1.0, 0.5),  # ignored
]
verts3 = [
   (1.5, 1.25),  # left, bottom
   (1.5, 1.5),  # left, top
   (2.0, 1.5),  # right, top
   (2.0, 1.25),  # right, bottom
   (1.5, 1.25),  # ignored
]
codes = [
    Path.MOVETO,
    Path.LINETO,
    Path.LINETO,
    Path.LINETO,
    Path.CLOSEPOLY,
]

path = Path(verts1, codes)

# fig, ax = plt.subplots()
patch = patches.PathPatch(path, facecolor='red', lw=2)
ax_mouse.add_patch(patch)
path = Path(verts2, codes)
patch = patches.PathPatch(path, facecolor='red', lw=2)
ax_mouse.add_patch(patch)
path = Path(verts3, codes)
patch = patches.PathPatch(path, facecolor='red', lw=2)
ax_mouse.add_patch(patch)

from scipy.spatial import ConvexHull, convex_hull_plot_2d
# manually making convex hull
convexhull1 = np.array([
   (0.0, 0.0),  # left, bottom
   (0.0, 2.0),  # left, top
   (0.5, 0.0),  # right, top
   (0.5, 2.0),  # right, bottom
])
convexhull2 = np.array([
   (0.35, 0.5),  # left, bottom
   (0.35, 2.0),  # left, top
   (1.0, 0.5),  # right, top
   (1.0, 2.0),  # right, bottom
])
convexhull3 = np.array([
   (0.85, 1.25),  # left, bottom
   (0.85, 2.0),  # left, top
   (1.5, 1.25),  # right, top
   (1.5, 2.0),  # right, bottom
])

hull1 = ConvexHull(convexhull1)
plt.plot(convexhull1[:,0], convexhull1[:,1], 'o')
hull2 = ConvexHull(convexhull2)
plt.plot(convexhull2[:,0], convexhull2[:,1], 'o')
hull3 = ConvexHull(convexhull3)
plt.plot(convexhull3[:,0], convexhull3[:,1], 'o')
for simplex in hull1.simplices:
    plt.plot(convexhull1[simplex, 0], convexhull1[simplex, 1], 'k-',color ='blue')
for simplex in hull2.simplices:
    plt.plot(convexhull2[simplex, 0], convexhull2[simplex, 1], 'k-',color='orange')
for simplex in hull3.simplices:
    plt.plot(convexhull3[simplex, 0], convexhull3[simplex, 1], 'k-',color='green')

fig_dev = plt.figure(2)
ax_dev = fig_dev.add_subplot(111)
ax_dev.set_xlim(-2,3) 
ax_dev.set_ylim(-2,3)
ax_dev.grid(color='gray', linestyle='--', linewidth=1)
ax_dev.set_title('derivative')

in_list_Chull=[]
in_list_Chull.append(hull1)
in_list_Chull.append(hull2)
in_list_Chull.append(hull3)
print in_list_Chull[0].equations
linebuilder = LineBuilder(line,hull1,hull2,in_list_Chull)




plt.show()

