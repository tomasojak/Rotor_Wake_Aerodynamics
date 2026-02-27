import numpy as np
import pandas as pd

# We want a handy way to load airfoil data from a file (or keep local capability to define it in code)
class Airfoil:
	"""
	Manages airfoil aerodynamic data including lift, drag, and moment coefficients.
	"""

	# Loads the values from an external file
	def __init__(self, filename):
		"""
		Load airfoil polar data from a CSV file.
		
		Args:
			filename: Path to CSV file containing columns 'Alfa', 'Cl', 'Cd', 'Cm'
			          where Alfa is in degrees
		"""
		self.data = pd.read_csv(filename)
		# Input in CSV is in degrees; convert once so internals use radians.
		self.alpha = np.deg2rad(self.data['Alfa'].values)
		self.cl = self.data['Cl'].values
		self.cd = self.data['Cd'].values
		self.cm = self.data['Cm'].values

	def lookup(self, alpha, variable: str):
		"""
		Perform linear interpolation to find aerodynamic coefficient at given angle of attack.
		
		Args:
			alpha: Angle of attack in radians
			variable: Coefficient to lookup - 'cl', 'cd', or 'cm'
			
		Returns:
			Interpolated coefficient value
			
		Raises:
			ValueError: If variable is not 'cl', 'cd', or 'cm'
		"""
		if variable == 'cl':
			return np.interp(alpha, self.alpha, self.cl)
		elif variable == 'cd':
			return np.interp(alpha, self.alpha, self.cd)
		elif variable == 'cm':
			return np.interp(alpha, self.alpha, self.cm)
		else:
			raise ValueError("Variable must be 'cl', 'cd', or 'cm'")
		
	def lookupAll(self, alpha) -> tuple:
		"""
		Lookup all aerodynamic coefficients (cl, cd, cm) at once.
		
		Args:
			alpha: Angle of attack in radians
			
		Returns:
			Tuple of (cl, cd, cm) values
		"""
		return self.lookup(alpha, 'cl'), self.lookup(alpha, 'cd'), self.lookup(alpha, 'cm')

class Geometry:
	"""
	Class to hold all the properties of the problem, including geometric, aerodynamic, and fluid properties.
	"""
	def __init__(self):
		self.tipRadius = 50.0  # m
		self.bladeStart = 0.2 # -
		self.bladeEnd = 1 # -
		self.bladeNumber = 3
		self.dimensionlessRadialPosition = lambda radius: radius / self.tipRadius
		# Aerodynamic    
		self.freeStreamVelocity = 10.0 # m/s
		self.tipSpeedRatio = 8.0
		# Load airfoil data
		self.airfoil = Airfoil('AirfoilPolarPlot.csv')
		self.bladeTwist = lambda radius: 14.0 * (1 - self.dimensionlessRadialPosition(radius)) # deg
		self.bladePitch = -2.0 # deg
		self.chordFunction = lambda radius: 3.0 * (1 - self.dimensionlessRadialPosition(radius)) + 1.0 # m
	
	@property
	def rotationalSpeed(self):
		"""Rotational speed in rad/s, computed from tip speed ratio."""
		return self.tipSpeedRatio * self.freeStreamVelocity / self.tipRadius

class IterationSolution:
	"""
	Class to hold the solution of a single iteration for a blade element.
	"""
	def __init__(self, radius, a, aPrime, iterations, precision, prandtlCorrection):
		self.radius = radius
		self.a = a
		self.aPrime = aPrime
		self.iterations = iterations
		self.precision = precision
		self.converged = precision < 1e-6 # This should be consistent with the tolerance used in the solver
		self.prandtlCorrection = prandtlCorrection
		
class Result:
	"""
	Class to hold the final results of the BEM solver, including thrust, torque, and coefficient of power.
	"""
	def __init__(self, radius, dT, dQ, cP, totalThrust, totalTorque):
		self.radius:np.array = radius
		self.dT:np.array = dT
		self.dQ:np.array = dQ
		self.cP:np.array = cP
		self.totalThrust:float = totalThrust
		self.totalTorque:float = totalTorque

class SolverData:
	"""
	Class to hold the properties and convergence history of the BEM solver.
	"""

	def __init__(self):

		self.geometry = Geometry()

		# Fluid properties for air at sea level
		self.airDensity = 1.225 # kg/m^3
		self.airViscosity = 1.81e-5 # kg/m/s

		self.elementSolutions: list[IterationSolution] = []
		# NOTE: Maybe this should be kept separately from the solution
		self.maxIterations = 100
		self.tolerance = 1e-6
		self.relaxation = 0.5
		self.elementCount = 10

		self.initialGuess = {
			"a": 0.3,
			"aPrime": 0.01
			}
		
		self.result = Result(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
	
	def setParameters(self, maxIterations, tolerance, relaxation, elementCount):
		"""
		Configure solver parameters for the BEM iterations.
		
		Args:
			maxIterations: Maximum number of iterations per radial station
			tolerance: Convergence tolerance for induction factors
			relaxation: Relaxation factor (0-1) for iterative updates
			elementCount: Number of radial blade elements to solve
		"""
		self.maxIterations = maxIterations
		self.tolerance = tolerance
		self.relaxation = relaxation
		self.elementCount = elementCount

	def addSolutionIteration(self, iterationResult:IterationSolution):
		"""
		Add a converged solution for a blade element to the solution history.
		
		Args:
			iterationResult: IterationSolution object containing converged values
		"""
		self.elementSolutions.append(iterationResult)

	def getElementSolution(self, radius):
		"""
		Retrieve the solution for a specific radial position.
		
		Args:
			radius: Radial position to search for (m)
			
		Returns:
			IterationSolution object if found, None otherwise
		"""
		for element in self.elementSolutions:
			if element.radius == radius:
				return element
		return None # Not found