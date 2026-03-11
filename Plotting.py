import numpy as np
import matplotlib.pyplot as plt
import SolverTools as st

def plot_bem_results(radius, elementSolutions):
	"""
	Plot the main BEM results including thrust/torque distributions and convergence metrics.

	Args:
		radius: Array of radial positions
		elementSolutions: List of IterationSolution objects containing convergence data
	"""

	convergencePrecision = np.array([element.precision for element in elementSolutions])
	convergenceIterations = np.array([element.iterations for element in elementSolutions])
	prandtlCorrection = np.array([element.prandtlCorrection for element in elementSolutions])
	axialInduction = np.array([element.a for element in elementSolutions])

	# Create our multi-subplot figure
	fig, axs = plt.subplots(2, 2, figsize=(14, 10))

	# Convergence precision plot
	axs[0, 0].plot(radius, convergencePrecision, marker='o')
	axs[0, 0].set_title('Convergence Precision vs Radius')
	axs[0, 0].set_xlabel('Radius (m)')
	axs[0, 0].set_ylabel('Convergence Precision')
	axs[0, 0].set_yscale('log')
	axs[0, 0].grid(True)

	# Iteration count plot
	axs[0, 1].plot(radius, convergenceIterations, marker='o', color='orange')
	axs[0, 1].set_title('Iteration Count vs Radius')
	axs[0, 1].set_xlabel('Radius (m)')
	axs[0, 1].set_ylabel('Iteration Count')
	axs[0, 1].grid(True)

	# Prandtl correction plot
	axs[1, 0].plot(radius, prandtlCorrection, marker='o', color='green')
	axs[1, 0].set_title('Prandtl Correction vs Radius')
	axs[1, 0].set_xlabel('Radius (m)')
	axs[1, 0].set_ylabel('Prandtl Correction')
	axs[1, 0].grid(True)

	# Axial induction factor plot
	axs[1, 1].plot(radius, axialInduction, marker='o', color='red')
	axs[1, 1].set_title('Axial Induction Factor (a) vs Radius')
	axs[1, 1].set_xlabel('Radius (m)')
	axs[1, 1].set_ylabel('Axial Induction Factor (a)')
	axs[1, 1].grid(True)

	plt.tight_layout()

def plot_pitch_variations(radiusGrid, pitchGrid, cPMap):
	"""
	Plot the coefficient of power (cP) as a function of radius and pitch angle, and identify the maximum cP for each radius.
	Args:
	radiusGrid: 1D array of radius values
	pitchGrid: 1D array of pitch angle values
	cPMap: 2D array of cP values with shape (len(pitchGrid), len(radiusGrid))
	"""

	max_indices = np.argmax(cPMap, axis=0)  # Find max pitch index for each radius
	max_pitches = pitchGrid[max_indices]  # Get the pitch values at those indices

	# 3D surface plot of cP
	fig_surface = plt.figure(figsize=(12, 8))
	ax_surface = fig_surface.add_subplot(111, projection='3d')
	R, P = np.meshgrid(radiusGrid, pitchGrid)
	ax_surface.plot_surface(R, P, cPMap, cmap='viridis', edgecolor='none')
	# Also plot the maximum cP points on the surface at z = cP values for better visualization
	max_cP_values = cPMap[max_indices, np.arange(len(radiusGrid))]
	ax_surface.scatter(radiusGrid, max_pitches, max_cP_values, c='r', marker='*', s=100, label='Maximum cP per radius')
	ax_surface.set_title('Coefficient of Power (cP) Surface Plot')
	ax_surface.set_xlabel('Radius (m)')
	ax_surface.set_ylabel('Pitch (deg)')
	ax_surface.set_zlabel('Coefficient of Power (cP)')
	ax_surface.legend(loc='best')
	plt.tight_layout()

def ShowPlots():
	"""
	Display all created matplotlib figures.
	"""
	plt.show()

##### Task d #####
def PlotResults(solution: st.SolverData):
	"""
	Plot:
	- a, a' distributions along the blade radial position
	- angle of attack distribution along the blade raidal position
	- inflow angle distribution along the blade radial position
	- Ct, Cn, Cq distribution along the blade radial position
	"""

	tipRadius = solution.geometry.tipRadius
	radius = np.array([element.radius for element in solution.elementSolutions])

	radialPosition = radius / tipRadius

	# Plotting the induction factors
	inductionFactors = np.array([element.a for element in solution.elementSolutions])
	inductionFactorsPrime = np.array([element.aPrime for element in solution.elementSolutions])


	plt.figure()
	plt.plot(radialPosition, inductionFactors, label='Axial Induction Factor (a)')
	plt.plot(radialPosition, inductionFactorsPrime, label='Tangential Induction Factor (a\')')
	plt.xlabel('Radial Position (r/R)')
	plt.ylabel('Induction Factor')
	plt.legend()
	plt.grid(True)

	# Plotting the angle of attack and inflow angle
	angleOfAttack = np.array([element.angleOfAttack for element in solution.elementSolutions])
	inflowAngle = np.array([element.inflowAngle for element in solution.elementSolutions])

	plt.figure()
	plt.plot(radialPosition, angleOfAttack, label='Angle of Attack (alpha)')
	plt.plot(radialPosition, inflowAngle, label='Inflow Angle (phi)')
	plt.xlabel('Radial Position (r/R)')
	plt.ylabel('Angle (degrees)')
	plt.legend()
	plt.grid(True)

	# Plotting the force coefficients
	Ct = np.array([element.dCt for element in solution.elementSolutions])
	Cn = np.array([element.dCn for element in solution.elementSolutions])
	Cq = np.array([element.dCq for element in solution.elementSolutions])

	plt.figure()
	plt.plot(radialPosition, Ct, label='Thrust Coefficient (dCt)')
	plt.plot(radialPosition, Cn, label='Normal Force Coefficient (dCn)')
	plt.plot(radialPosition, Cq, label='Torque Coefficient (dCq)')
	plt.xlabel('Radial Position (r/R)')
	plt.ylabel('Coefficient')
	plt.legend()
	plt.grid(True)