# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 09:58:58 2021

@author: Martin Alejandro Cervantes Balderrama
"""
import math 
import numpy as np
import matplotlib.pyplot as plt
import time
import sim
from fuzzy_art import FuzzyARTMAP as FAM
 
def connect(port):
    # Connection with Coppelia
    sim.simxFinish(-1)
    clientID = sim.simxStart('127.0.0.1', port, True, True, 2000, 5)
    if clientID == 0: print("Successfully conected to port ", port)
    else: print("Connection error")
    return clientID

clientID = connect(19999)

q0 = np.array([0, math.pi/2])
dq0 = np.array([0, 0])
h = 0.01
tf = 10+h
t = np.arange(0, tf, h)
xc_plt = np.zeros((len(t), 2))
fe_plt = np.zeros((len(t), 2))
fd_plt = np.zeros((len(t), 1))

#Coppelia Data
returnCode, joint1 = sim.simxGetObjectHandle(clientID, "joint1", 
                                             sim.simx_opmode_blocking)
returnCode, joint2 = sim.simxGetObjectHandle(clientID, "joint2", 
                                             sim.simx_opmode_blocking)
returnCode, forceSensorHandle = sim.simxGetObjectHandle(clientID, "forceSensor", 
                                             sim.simx_opmode_blocking)

#Path's points
f = open("C:\\Users\\nitra\\OneDrive\\Escritorio\\Tesis\\V-REP\\path.txt", "r")
posPointsTxt = f.readlines()
arrPos = np.zeros((len(posPointsTxt)+1, 2))
j = 0

for i in posPointsTxt:
    posPoints = i.split(",")
    posPoints.pop() 
    
    x_0p = float(posPoints[0])
    y_0p = float(posPoints[2])-.425
    xp = np.array([x_0p, y_0p])
    arrPos[j,:] = xp
    j = j+1

arrPos[j,:] = xp
f.close()


# Training values for motion on 'x' and 'y' directions in one path from botton to 
# top and right to left
inputs_ARTa = np.array([[0, 1.5e-4],
                        [1e-5, 1.5e-4],
                        [-1e-5, 1.5e-4],
                        [-1.5e-4, 0],
                        [-1.5e-4, 1e-5],
                        [-1.5e-4, -1e-5]])

inputs_ARTa_norm = (inputs_ARTa - inputs_ARTa.min())/(
        inputs_ARTa - inputs_ARTa.min()).max()

print(inputs_ARTa_norm)

# Clasification of values for motion on 'x' and 'y' directions in one path from botton to 
# top and left to right
inputs_ARTb = np.array([0, 0, 0, 1, 1, 1])

net = FAM(1.0, 0.01, 0.5, 0.01,True)
net.train(inputs_ARTa_norm, inputs_ARTb)

inputs_Test = np.zeros((len(arrPos)-1,2))

for i in range(0, len(arrPos)-1):

    inputs_Test[i,:] = np.around(arrPos[i+1,:], decimals=5)-np.around(
                                                        arrPos[i,:], decimals=5)  
    if i==len(arrPos)-2:
        inputs_Test[i] = inputs_Test[i-1]

inputs_Test_norm = (inputs_Test - inputs_Test.min())/(
                        inputs_Test - inputs_Test.min()).max()

labelTest = net.test(inputs_Test_norm).astype(int)

#Definig S matrix for every point in path
arrS = np.zeros((len(labelTest),2,2))

for i in range(0, len(labelTest)):
    if labelTest[i] == 0:
        arrS[i,:,:] = np.array([[0,0],
                                [0,1]])
    if labelTest[i] == 1:
        arrS[i,:,:] = np.array([[1,0],
                                [0,0]])  

time.sleep(2)

def dynamicRobot(q, dq):
    m1 = 1.2
    m2 = 1
    l1 = .3
    l2 = .24 
    cf1 = 0.5
    cf2 = 0.5
    g = 9.81
    tao = np.array([0, 0])
    yk_p_prev = np.array([0, 0])
    yk_f_prev2 = np.array([0, 0])  

    for i in range(0, len(t)):
        time.sleep(h)
        returnCode = sim.simxSetJointTargetPosition(clientID, joint1, q[0], 
                                            sim.simx_opmode_oneshot)
        returnCode = sim.simxSetJointTargetPosition(clientID, joint2, q[1], 
                                            sim.simx_opmode_oneshot)
         
        c1 = math.cos(q[0])
        c2 = math.cos(q[1])
        s1 = math.sin(q[0])
        s2 = math.sin(q[1])
        c12 = math.cos(q[0]+q[1])
        s12 = math.sin(q[0]+q[1])  
        
#       Cartesian Velocities
#       Jacobian exprese in frame 0
        J = np.array([[-l1*s1-l2*s12, -l2*s12],
                      [l1*c1+l2*c12, l2*c12]])    
      
        dx = J@dq
        
#       Kinematic 
        T_03 = np.array([[c12, -s12, l1*c1+l2*c12],[s12, c12, l1*s1+l2*s12],
                         [0, 0, 1]])

        xc = np.array([T_03[0:2,2]])
        xc_plt[i,:] = xc       
       
        x_d =  arrPos[i,:]
        if i < 1000: 
            Sp = arrS[i,:,:]

        if i == 1000: 
            Sp = arrS[i-1,:,:]            
        Sf = np.identity(2)-Sp
        
#       Force Control      
        fd = np.array([15, 15])  
        fd_plt[i] = fd[1]
        returnCode, state, forceVector, torqueVector = sim.simxReadForceSensor(
                clientID, forceSensorHandle, sim.simx_opmode_blocking)
        
        fs = np.array([forceVector[0], forceVector[2]])
        fs = abs(fs)

        fe_plt[i,:] = fs

        kf = np.array([[.0003, 0],[0, .0003]])
        kif = np.array([[.002, 0],[0, .002]])
        xf = xc
        
        ef = fd-fs
        
        xf = ef@kf+yk_f_prev2@kif+xf # PI Control

#       Force Error Integration (Backward Rectangule Rule)
        T = h
        xk_f2 = ef
        yk_f2 = T*xk_f2+yk_f_prev2
        yk_f_prev2 = yk_f2
        
        xf = xf@Sf
        
        x_d = Sp@x_d
        x_d = x_d+xf
        
#       Position Control      
        dx_d = np.array([0.1, 0.1]) 
        
        ep = ei = x_d-xc        
        ed = dx_d-dx
        
        kp = np.array([[600, 0],[0, 600]])
        ki = np.array([[600, 0],[0, 600]])
        kd = np.array([[40, 0],[0, 40]])
        
        ep = ep@kp
        ei = ei@ki
        ed = ed@kd
        
        d2x_d = np.array([0, 0]) 
               
        fp = d2x_d+ep+yk_p_prev+ed # PID Control
        
#       Position Error Integration        
        T = h
        xk_p = ei
        yk_p = T*xk_p+yk_p_prev
        yk_p_prev = yk_p       

        f_pf = fp
        
#       Mx Matrix                
        M = np.array([[m2*l2**2+2*m2*l1*l2*c2+(m1+m2)*l1**2, m2*l2**2+m2*l1*l2*c2],
                       [m2*l1*l2*c2+m2*l2**2, m2*l2**2]])
        
        invJ = np.linalg.inv(J)
        invTransJ = np.transpose(invJ)
        
        Mx = invTransJ@M@invJ  
  
        A2 = f_pf@Mx 
        fd = fd@Sf 

#       Vx Matrix        
        V = np.array([-2*m2*l1*l2*s2*dq[0]*dq[1]-m2*l1*l2*s2*dq[1]**2,
                      m2*l1*l2*s2*dq[0]**2])
    
        # Jacobian exprese in frame 0
        dJ = np.array([[-l1*c1*dq[0]-l2*c12*(dq[0]+dq[1]), -l2*c12*(dq[0]+dq[1])],
                       [-l1*s1*dq[0]-l2*s12*(dq[0]+dq[1]), -l2*s12*(dq[0]+dq[1])]])    
        
        Vx = invTransJ@(V-M@invJ@dJ@dq)

#       Gx Matrix      
        G = np.array([m2*l2*g*c12+(m1+m2)*l1*g*c1,
                      m2*l2*g*c12])

        Gx = invTransJ@G 
        
        beta = Vx+Gx
        F = beta+A2
 
        Jt = np.transpose(J)
        tao = Jt@F[0]
        
#       Dynamic  
        invM = np.linalg.inv(M)         
        D = np.array([[cf1, 0], [0, cf2]])   
        
#       Improved Euler's Method
        d2q_prev = invM@(tao-V-(D@dq)-G)
        
#       Prediction
        dq_pred = dq+d2q_prev*h
        q_pred = q+dq_pred*h
        
        c1 = math.cos(q_pred[0])
        c2 = math.cos(q_pred[1])
        s2 = math.sin(q_pred[1])
        c12 = math.cos(q_pred[0]+q_pred[1])   
        
        M = np.array([[m2*l2**2+2*m2*l1*l2*c2+(m1+m2)*l1**2, m2*l2**2+m2*l1*l2*c2],
                       [m2*l1*l2*c2+m2*l2**2, m2*l2**2]])
        invM = np.linalg.inv(M)   
    
        V = np.array([-2*m2*l1*l2*s2*dq_pred[0]*dq_pred[1]-m2*l1*l2*s2*dq_pred[1]**2,
                      m2*l1*l2*s2*dq_pred[0]**2])   
    
        G = np.array([m2*l2*g*c12+(m1+m2)*l1*g*c1,
                      m2*l2*g*c12])

        d2q_post = invM@(tao-V-(D@dq_pred)-G)  
        
#       Correction
        dq_corr = dq+(d2q_prev+d2q_post)*(h/2)
        q_corr = q+(dq+dq_corr)*(h/2)
    
        dq = dq_corr
        q = q_corr
        
#   Graphs    
    plt.plot(t,xc_plt[:,0],color='b', linewidth=1.2)
    plt.plot(t,xc_plt[:,1],color='r', linewidth=1.2)
    plt.grid()
    plt.legend((r'$x$', r'$y$'),
           prop = {'size': 10}, loc='upper right')
    plt.xlabel(r'$time\ \ [s]$')
    plt.ylabel(r'$X\ \ [m]$')
    plt.title('Posiciones Cartesianas, h=0.01', fontsize=14)
    plt.xlim([0, tf])
    plt.show()  
    
    plt.plot(t,fd_plt[:],color='g', linestyle='dashdot', linewidth=1)
    plt.plot(t,fe_plt[:,0],color='b', linewidth=1.2)
    plt.plot(t,fe_plt[:,1],color='r', linewidth=1.2)
    plt.grid()
    plt.legend((r'$F_d$', r'$F_x$', r'$F_y$'),
           prop = {'size': 10}, loc='upper right')
    plt.xlabel(r'$time\ \ [s]$')
    plt.ylabel(r'$F\ \ [N]$')
    plt.title('Fuerzas de Contacto, h=0.01', fontsize=14)
    plt.xlim([0, tf])
    plt.show() 

dynamicRobot(q0, dq0)



    