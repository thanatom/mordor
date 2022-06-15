from numba import deferred_type, optional, prange, float32, float64, boolean, int64, njit
from numba.experimental import jitclass
from pynbody import array, units
import numpy as np

'''

Classes and functions to compute gravity through a kdtree.
These tools are based on pykdgrav (now pytreegrav, https://github.com/mikegrudic/pytreegrav)
Some functions have been added/adapted

'''

node_type = deferred_type()

spec = [
	('bounds', float64[:,:]),
	('size', float64),
	('delta', float64),
	('points', float64[:,:]),
	('masses', float64[:]),
	('Npoints', int64),
	('h', float64),
	('softening', float64[:]),
	('mass', float64),
	('COM', float64[:]),
	('IsLeaf', boolean),
	('HasLeft', boolean),
	('HasRight', boolean),
	('left', optional(node_type)),
	('right', optional(node_type)),
]

#########################################################################################################

@jitclass(spec)
class KDNode(object):
	def __init__(self, points, masses, softening):
		self.bounds = np.empty((3,2))
		self.bounds[0,0] = points[:,0].min()
		self.bounds[0,1] = points[:,0].max()
		self.bounds[1,0] = points[:,1].min()
		self.bounds[1,1] = points[:,1].max()
		self.bounds[2,0] = points[:,2].min()
		self.bounds[2,1] = points[:,2].max()

		self.softening = softening
		self.h = self.softening.max()
		
		self.size = max(self.bounds[0,1]-self.bounds[0,0],self.bounds[1,1]-self.bounds[1,0],self.bounds[2,1]-self.bounds[2,0])
		self.points = points
		self.Npoints = points.shape[0]
		self.masses = masses
		self.mass = np.sum(masses)
		self.delta = 0.
		if self.Npoints == 1:
			self.IsLeaf = True
			self.COM = points[0]
		else:
			self.IsLeaf = False
			self.COM = np.zeros(3)
			for k in range(3):
				for i in range(self.Npoints):
					self.COM[k] += points[i,k]*masses[i]
				self.COM[k] /= self.mass
				self.delta += (0.5*(self.bounds[k,1]+self.bounds[k,0]) - self.COM[k])**2
			self.delta = np.sqrt(self.delta)

		self.HasLeft = False
		self.HasRight = False        
		self.left = None
		self.right = None

	def GenerateChildren(self, axis):
		if self.IsLeaf:
			return 0
		x = self.points[:,axis]
		med = (self.bounds[axis,0] + self.bounds[axis,1])/2
		index = (x<med)
		if np.any(index):
			self.left = KDNode(self.points[index], self.masses[index], self.softening[index])
			self.HasLeft = True
		index = np.invert(index)
		if np.any(index):
			self.right = KDNode(self.points[index],self.masses[index], self.softening[index])
			self.HasRight = True
		self.points = np.empty((1,1))
		self.masses = np.empty(1)
		self.softening = np.empty(1)
		return 1

node_type.define(KDNode.class_type.instance_type)

#########################################################################################################

@njit
def ConstructKDTree(x, m, softening):
	if len(np.unique(x[:,0])) < len(x):
		raise Exception("Non-unique particle positions are currently not supported by the tree-building algorithm. Consider perturbing your positions with a bit of noise if you really want to proceed.")
	root = KDNode(x, m, softening)
	nodes = [root,]
	axis = 0
	divisible_nodes = 1
	count = 0
	while divisible_nodes > 0:
		N = len(nodes)
		divisible_nodes = 0
		for i in range(count, N): # loop through the nodes we spawned in the previous pass
			count += 1
			if nodes[i].IsLeaf:
				continue                
			else:
				generated_children = nodes[i].GenerateChildren(axis)
				divisible_nodes += generated_children
				if nodes[i].HasLeft:
					nodes.append(nodes[i].left)
				if nodes[i].HasRight:
					nodes.append(nodes[i].right)
					
		axis = (axis+1)%3
	return root

#########################################################################################################

@njit(fastmath=True)#([float64(float64,float64),float32(float32,float32)])
def ForceKernel(r, h):

	"""
	Returns the quantity (fraction of mass enclosed)/ r^3 for a cubic-spline mass distribution of compact support radius h. Used to calculate the softened gravitational force.

	Arguments:
	r - radius
	h - softening
	"""
	
	if r > h: return 1./(r*r*r)
	hinv = 1./h
	q = r*hinv
	if q <= 0.5:
		return (10.666666666666666666 + q*q*(-38.4 + 32.*q))*hinv*hinv*hinv
	else:
		return (21.333333333333 - 48.0 * q + 38.4 * q * q - 10.666666666667 * q * q * q - 0.066666666667 / (q * q * q))*hinv*hinv*hinv

#########################################################################################################

@njit(fastmath=True)
def PotentialKernel(r, h):

	"""
	Returns the equivalent of -1/r for a cubic-spline mass distribution of compact support radius h. Used to calculate the softened gravitational potential.

	Arguments:
	r - radius
	h - softening
	"""

	if h==0.:
		return -1./r
	hinv = 1./h
	q = r*hinv
	if q <= 0.5:
		return (-2.8 + q*q*(5.33333333333333333 + q*q*(6.4*q - 9.6))) * hinv
	elif q <= 1:
		return (-3.2 + 0.066666666666666666666 / q + q*q*(10.666666666666666666666 +  q*(-16.0 + q*(9.6 - 2.1333333333333333333333 * q)))) * hinv
	else:
		return -1./r

#########################################################################################################

def KDPotential(pos, m, softening=None, theta=0.5, tree=None):

	"""
	Returns the approximate gravitational potential for a set of particles with positions x and masses m.

	Arguments:
	pos -- shape (N,3) array of particle positions
	m -- shape (N,) array of particle masses

	Keyword arguments:
	softening -- shape (N,) array containing kernel support radii for gravitational softening
	tree -- optional pre-generated kd-tree: this can contain any set of particles, not necessarily the target particles at pos (default None)

	Parameters:
	theta -- cell opening angle used to control force accuracy; smaller is slower (runtime ~ theta^-3) but more accurate. (default 1.0, gives ~1% accuracy)

	Returns:
	Pynbody SimArray with the potentil energy
	"""

	if softening is None:
		softening = np.zeros_like(m)
	if tree is None: tree = ConstructKDTree(np.float64(pos), np.float64(m), np.float64(softening))
	result = np.zeros(len(m))

	pot = GetPotentialParallel(np.float64(pos), tree, theta=theta)

	pot = pot.view(array.SimArray)
	pot.units = units.G * m.units / pos.units

	return pot

#########################################################################################################

@njit(fastmath=True, parallel=True)
def BruteForcePotentialTarget(x_target, x_source, m_source, softening=None):

	"""
	Returns the exact gravitational potential in units of G due to a set of particles, at a set of positions that need not be the same as the particle positions.

	Arguments:
	x_target -- shape (N,3) array of positions where the potential is to be evaluated
	x_source -- shape (M,3) array of positions of gravitating particles
	m_source -- shape (N,) array of particle masses

	Keyword arguments:
	softening -- shape (M,) array containing kernel support radii for gravitational softening
	"""

	if softening is None: softening = np.zeros(x_target.shape[0])
	potential = np.zeros(x_target.shape[0])
	for i in prange(x_target.shape[0]):
		for j in range(x_source.shape[0]):
			dx = x_target[i,0]-x_source[j,0]
			dy = x_target[i,1]-x_source[j,1]
			dz = x_target[i,2]-x_source[j,2]
			r = np.sqrt(dx*dx + dy*dy + dz*dz)
			if r < softening[j]:
				potential[i] += m_source[j] * PotentialKernel(r, softening[j])
			else:
				if r>0: potential[i] -= m_source[j]/r

	return potential

#########################################################################################################

@njit(fastmath=True, parallel=True)
def BruteForceAccelTarget(x_target, x_source, m_source, softening=None):

	"""
	Returns the gravitational acceleration in units of G produced by the mass of a set of particles, at a set of positions that need not be the same as the particle positions.

	Arguments:
	x_target -- shape (N,3) array of positions where the potential is to be evaluated
	x_source -- shape (M,3) array of positions of gravitating particles
	m_source -- shape (N,) array of particle masses

	Keyword arguments:
	softening -- shape (M,) array containing kernel support radii for gravitational softening
	"""

	if softening is None: softening = np.zeros(x_target.shape[0])
	accel = np.zeros_like(x_target)
	for i in prange(x_target.shape[0]):
		for j in range(x_source.shape[0]):
			dx = x_target[i,0]-x_source[j,0]
			dy = x_target[i,1]-x_source[j,1]
			dz = x_target[i,2]-x_source[j,2]
			r = np.sqrt(dx*dx + dy*dy + dz*dz)
			if r < softening[j]:
				fac = -m_source[j] * ForceKernel(r, softening[j])
			else:
				fac = -m_source[j]/(r*r*r)
			accel[i,0] += fac*dx
			accel[i,1] += fac*dy
			accel[i,2] += fac*dz
	return accel

#########################################################################################################

@njit(fastmath=True)
def PotentialWalk(pos, node, phi, theta=0.5):

	"""
	Returns the gravitational field in units of G at position x by performing the Barnes-Hut treewalk using the provided KD-tree node

	Arguments:
	pos - (3,) array containing position of interest
	node - KD-tree to walk
	phi - temporary potential

	Parameters:
	theta - cell opening angle used to control force accuracy; smaller is slower (runtime ~ theta^-3) but more accurate. (1.0, gives ~1 percent accuracy)
	"""
	
	dx = node.COM[0]-pos[0]
	dy = node.COM[1]-pos[1]
	dz = node.COM[2]-pos[2]
	r = np.sqrt(dx*dx + dy*dy + dz*dz)
	if node.IsLeaf:
		if r>0:
			phi += node.mass * PotentialKernel(r, node.h)
	elif r > max(node.size/theta, node.h+node.size):
		phi -= node.mass/r
	else:
		if node.HasLeft:
			phi = PotentialWalk(pos, node.left, phi, theta=theta)
		if node.HasRight:
			phi = PotentialWalk(pos, node.right, phi, theta=theta)
	return phi

#########################################################################################################

@njit(fastmath=True)
def ForceWalk(pos, node, g, softening=0.0, theta=0.5):

	"""
	Returns the gravitational field at position pos by performing the Barnes-Hut treewalk using the provided KD-tree node

	Arguments:
	pos - (3,) array containing position of interest
	node - KD-tree to walk
	g - (3,) array containing initial value of the gravitational field, used when adding up the contributions in recursive calls

	Parameters:
	softening - softening radius of the particle at which the force is being evaluated - needed if you want the short-range force to be momentum-conserving
	theta - cell opening angle used to control force accuracy; smaller is slower (runtime ~ theta^-3) but more accurate. (1.0, gives ~1percent accuracy)
	"""

	dx = node.COM[0]-pos[0]
	dy = node.COM[1]-pos[1]
	dz = node.COM[2]-pos[2]
	r = np.sqrt(dx*dx + dy*dy + dz*dz)
	add_accel = False
	fac = 0
	if r>0:
		if node.IsLeaf:
			add_accel = True
			if r < max(node.h, softening):
				fac = node.mass * ForceKernel(r, max(node.h, softening))
			else:
				fac = node.mass/(r*r*r)
		elif r > max(node.size/theta + node.delta, max(node.h,softening)+node.size):
			add_accel = True  
			fac = node.mass/(r*r*r)

	if add_accel:
		g[0] += dx*fac
		g[1] += dy*fac
		g[2] += dz*fac
	else:
		if node.HasLeft:
			g = ForceWalk(pos, node.left, g, softening=softening, theta=theta)
		if node.HasRight:
			g = ForceWalk(pos, node.right, g, softening=softening, theta=theta)
	return g

#########################################################################################################

@njit(parallel=True, fastmath=True)
def GetPotentialParallel(pos, tree, theta=0.5):
	result = np.empty(pos.shape[0])
	for i in prange(pos.shape[0]):
		result[i] = PotentialWalk(pos[i], tree, 0., theta=theta)
	return result

#########################################################################################################

@njit(parallel=True, fastmath=True)
def GetAccelParallel(pos, tree, softening, theta=0.5):
	if softening is None: softening = np.zeros(len(pos), dtype=np.float64)    
	result = np.empty(pos.shape)
	for i in prange(pos.shape[0]):
		result[i] = ForceWalk(pos[i], tree, np.zeros(3), softening=softening[i], theta=theta)
	return result
