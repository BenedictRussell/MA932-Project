import numpy as np
from scipy.integrate import odeint
from scipy import stats
from scipy.stats import uniform
import matplotlib.pyplot as plt
from matplotlib  import cm
import matplotlib as mpl
import math


def main():

	# Simulation parameters
	v0           = 1.0      # velocity
	eta          = 0.5      # random fluctuation in angle (in radians)
	L            = 10       # size of box
	R            = 1        # interaction radius
	dt           = 0.2      # time step
	Nt           = 200      # number of time steps
	N            = 10      # number of balls
	
	# Initialize
	np.random.seed(17)      # set the random number generator seed

	# initial positions
	x = np.random.rand(N)*L
	y = np.random.rand(N)*L
	
	# initial velocities
	theta = 2 * np.pi * np.random.rand(N)
	vx = v0 * np.cos(theta)
	vy = v0 * np.sin(theta)
	
    # Arrays to store positions and orientations
	x_time = np.array(np.zeros((Nt,N)))
	y_time = np.array(np.zeros((Nt,N)))
	theta_time = np.array(np.zeros((Nt,N)))
	x_time[0,:] = x

	# Simulation Main Loop
	for i in range(Nt):

		# move
		x += vx*dt
		y += vy*dt
		
		# apply periodic BCs
		x = x % L
		y = y % L
		
        #
		dists = (x-5)**2 + (y-5)**2
		
		# find mean angle of neighbors within R
		mean_theta = np.copy(theta)
		theta_rad = np.zeros(N)

		for b in range(N):
			neighbors = (x-x[b])**2+(y-y[b])**2 < R**2
			sx = np.sum(np.cos(theta[neighbors]))
			sy = np.sum(np.sin(theta[neighbors]))
			mean_theta[b] = np.arctan2(sy, sx)
	
			
		# add random perturbations
		theta = mean_theta + eta*(np.random.rand(N)-0.5)
		#theta -= theta_rad
		
		# update velocities
		vx = v0 * np.cos(theta)
		vy = v0 * np.sin(theta)

		# append
		x_time[i,:] = x
		y_time[i,:] = y
		theta_time[i,:] = theta
	
	return x_time, y_time, theta_time

def dynamic_gen():

    # Simulation parameters
	v0           = 5.0      # velocity
	eta          = 0.5      # random fluctuation in angle (in radians)
	L            = 100       # size of box
	R            = 1        # interaction radius
	dt           = 0.1      # time step
	N            = 2      # number of balls

    #
	x,y = np.random.uniform(low=40, high=60, size=(2, N))
	#x,y = np.zeros(2)+50
	vx,vy = 0.1*np.random.uniform(low=1, high=1, size=(2, N))
	theta = 2 * np.pi * np.random.rand(N)
	vx = np.cos(theta)
	vy = np.sin(theta)
	dW = np.ones(N)	

	yield x,y,theta, dW
	r = 5
	k = 40
	
    #
	while True:
		#print(dW)
		x += dW*vx*dt; x = x % L
		y += dW*vy*dt; y = y % L
	
		theta += 0.5*(np.random.rand(N)-0.5)
		
		# dx[i,j] = x[i]-x[j]
		dx = x[:,None]-x[None,:]; dx = (dx) % L 
		dy = y[:,None]-y[None,:]; dy = (dy) % L
		
        #
		dr = np.hypot(dx,dy)
		dr = np.ma.masked_where( ~((0<dr)&(dr<r)), dr )
        
        # size of force if less than or equal to radius
		force = k*(2*r-dr)
		
        #
		dW = np.sqrt((x-50)**2+(y-50)**2)
		R = 50
		dW = np.where(R-dW <= r, -1, 1)
		
		#theta = np.where(R-dW <= 2*r,-theta,theta)
		#print(theta)
			
		vx = (v0 *np.cos(theta) + np.ma.sum(force*dx/dr,axis=1))*dt 
		vy = (v0 *np.sin(theta) + np.ma.sum(force*dy/dr,axis=1))*dt 

		#print(dW, vx)
		yield x,y,theta, dW

def scatter_t(x,y,theta,t):
    
    fig = plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
    norm = mpl.colors.Normalize(vmin=0, vmax=2*np.pi)
    cmap = cm.gist_rainbow
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    temp = m.to_rgba(np.mod(theta[t,:],2*np.pi))
    plt.scatter(x[t,:],y[t,:], c = temp, s = 100, alpha = 0.9, marker = 'o',edgecolors='none',cmap = cm.gist_rainbow)
    plt.xlim(0,10)
    plt.ylim(0,10)

    #theta = np.linspace( 0 , 2 * np.pi , 150 )
 
    #radius = 100
    
    #a = radius * np.cos( theta )
    #b = radius * np.sin( theta )
    
    #plt.plot( a, b )
	
def scatter(x,y,theta):
    
    fig = plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
    norm = mpl.colors.Normalize(vmin=0, vmax=2*np.pi)
    cmap = cm.gist_rainbow
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    temp = m.to_rgba(np.mod(theta,2*np.pi))
    plt.scatter(x,y, c = temp, s = 400, alpha = 0.9, marker = 'o',edgecolors='none',cmap = cm.gist_rainbow)
    plt.xlim(0,100)
    plt.ylim(0,100)
	
    theta = np.linspace( 0 , 2 * np.pi , 150 )
 
    radius = 50
    
    a = 50+ radius * np.cos( theta )
    b = 50+ radius * np.sin( theta )
    
    plt.plot( a, b )