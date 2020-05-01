import os
import sys

import numpy as np

import crocoddyl
import example_robot_data
import pinocchio

import lcmaes 

WITHDISPLAY = 'display' in sys.argv or 'CROCODDYL_DISPLAY' in os.environ
WITHPLOT = 'plot' in sys.argv or 'CROCODDYL_PLOT' in os.environ

# WITHDISPLAY = 'display'# in sys.argv or 'CROCODDYL_DISPLAY' in os.environ
WITHPLOT ='EMPTY'
crocoddyl.switchToNumpyMatrix()

# Loading the anymal model
anymal = example_robot_data.loadANYmal()

display = crocoddyl.GepettoDisplay(anymal, 4, 4)

# Defining the initial state of the robot
print ('model name: ' + anymal.model.name)
        
data = anymal.model.createData()

q = anymal.model.referenceConfigurations['standing'].copy()
pinocchio.forwardKinematics(anymal.model,data,q)
display.robot.display(q)

#---------------------------------------------------
# display setup

frameCoMTrajNames = []
frameCoMTrajNames.append(str(anymal.model.getFrameId('root_joint')))
frameTrajColor = {}
frameTrajLineWidth = 5
frameTrajColor = list(np.hstack([np.random.choice(range(256), size=3) / 256., 1.]))

# add curve
for key in frameCoMTrajNames:
    frameName = display.frameTrajGroup + "/" + key
    display.robot.viewer.gui.addCurve(frameName, [np.array([0., 0., 0.]).tolist()] * 2, frameTrajColor)
    display.robot.viewer.gui.setCurveLineWidth(frameName, frameTrajLineWidth)
    if display.fullVisibility:
        display.robot.viewer.gui.setVisibility(frameName, "ALWAYS_ON_TOP")

# def calcCoMmodel(T,cop_dpl,ptsTime):
#     #--heuristic
#     height = data.oMi[1].translation[2]
#     com_p0 = pinocchio.centerOfMass(anymal.model,data,q).T
#     com_v0 = np.array([0.01,0.0,0],dtype=float)
#     #-- new cop
#     com_p0[0,2] = 0
#     cop_p0 = com_p0

#     k = cop_dpl/T
#     print anymal.model.gravity
#     w = np.sqrt(9.81/height)
#     beta1 = (com_p0 - cop_p0)/2 + (com_v0*T - cop_dpl)/(2*w*T)
#     beta2 = (com_p0 - cop_p0)/2 - (com_v0*T - cop_dpl)/(2*w*T)

#     cnt = 0
#     pos_comH = np.zeros((T,3),dtype=float)
#     for t in (ptsTime):
#         exp1 = np.exp(w*t)
#         exp2 = np.exp(-w*t)
#         pos = np.multiply(beta1,exp1) + np.multiply(beta2,exp2) + cop_p0 + np.multiply(k,t)
#         pos_comH[cnt] = pos
#         cnt = cnt + 1        
#     return pos_comH

def calcCoMmodel(T,cop_dpl,com_p0,com_v0,cop_p0):
    #--heuristic
    height = data.oMi[1].translation[2]

    # cop 
    new_cop_p = calcCoPmodel(T,cop_dpl)

    w = np.sqrt(9.81/height)
    beta1 = (com_p0 - cop_p0)/2 + (com_v0*T - cop_dpl)/(2*w*T)
    beta2 = (com_p0 - cop_p0)/2 - (com_v0*T - cop_dpl)/(2*w*T)

    exp1 = np.exp(w*T)
    exp2 = np.exp(-w*T)
    pos = np.multiply(beta1,exp1) + np.multiply(beta2,exp2) + new_cop_p
        
    return pos

def calcCoPmodel(T,cop_dpl):
    k = cop_dpl/T
    pos = cop_p0 + np.multiply(k,T)
    return pos

# creating control parameter
# fixed 6 phase [ stance, swing, swing, stance, swing, swing] 
U_cp = []
for i in range(6):
    cp = []
    if (i == 0 or i == 3): # stance phase [duration, dpl]
        for j in range(2):
            cp.append(1e-3)
    else :                 # swing phase [duration, dpl, foot dpl]
        for j in range(3):
            cp.append(1e-3)

    U_cp.append(cp)
print U_cp

# you should put initial values inside U_cp

# Updating State From U
cur_com_p0 = pinocchio.centerOfMass(anymal.model,data,q).T
cur_com_v0 = np.array([0.01,0.0,0],dtype=float)
cur_cop_p0 = cur_com_p0
for cp in U_cp:

    # control parameter
    # cp[0] = T : duration
    # cp[1] = cop_delta : displacement of CoP
    # cp[2] = foot dpl for swing phase

    T = cp[0]
    cur_cop_dpl = cp[1]

    # state
    cur_com_pos = calcCoMmodel(T, cur_cop_dpl, cur_com_p0, cur_com_v0, cur_cop_p0)
    cur_com_vel = cur_com_pos / T
    cur_cop_p0 = calcCoPmodel(T,cur_cop_dpl)

    # cost function


    # new cur
    cur_com_v0 = cur_com_vel
    cur_com_p0 = cur_com_pos



    


    



# #plotting sampled points
# numSamples = 10; fNumSamples = float(numSamples)
# ptsTime = [ (float(t) * 1 / fNumSamples) for t in range(numSamples+1)]
# print ptsTime
# cop_delta0 = np.array([0.01,0,0],dtype=float)
# pos_comH = calcCoMmodel(numSamples+1,cop_delta0,ptsTime)

# #display
# ps = {fr: [] for fr in frameCoMTrajNames}
# for key, p in ps.items():
#     for pos in pos_comH:
#         p.append(np.asarray(pos).reshape(-1).tolist())
# # set curve points 
# for key, p in ps.items():
#     display.robot.viewer.gui.setCurvePoints(display.frameTrajGroup + "/" + key, p)

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()

# ax = fig.add_subplot(111)

# # Major ticks every 20, minor ticks every 5
# major_ticks = np.arange(0, 101, 20)
# minor_ticks = np.arange(0, 101, 5)

# ax.set_xticks(major_ticks)
# ax.set_xticks(minor_ticks, minor=True)
# ax.set_yticks(major_ticks)
# ax.set_yticks(minor_ticks, minor=True)

# # And a corresponding grid
# ax.grid(which='both')

# # Or if you want different settings for the grids:
# ax.grid(which='minor', alpha=0.2)
# ax.grid(which='major', alpha=0.5)

# # ax.plot(pos_comH[:,0],pos_comH[:,1],0)
# # ax.scatter(pos_comH[:,0],pos_comH[:,1],0)

# plt.plot(pos_comH[:,0],pos_comH[:,1])
# plt.scatter(pos_comH[:,0],pos_comH[:,1],marker='o',facecolor='red')
# for i, txt in enumerate(ptsTime):
#     ax.annotate(txt, (pos_comH[i,0], pos_comH[i,1]))

# ax.set_xlabel('x')
# ax.set_ylabel('y')
# # ax.set_zlabel('time')
# plt.show()





