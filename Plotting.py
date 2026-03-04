import numpy as np
import matplotlib.pyplot as plt

def plot_bem_results(radius, dT, dQ, elementSolutions):
	"""
	Plot the main BEM results including thrust/torque distributions and convergence metrics.

	Args:
		radius: Array of radial positions
		dT: Array of differential thrust values
		dQ: Array of differential torque values
		elementSolutions: List of IterationSolution objects containing convergence data
	"""

	convergencePrecision = np.array([element.precision for element in elementSolutions])
	convergenceIterations = np.array([element.iterations for element in elementSolutions])
	prandtlCorrection = np.array([element.prandtlCorrection for element in elementSolutions])
	axialInduction = np.array([element.a for element in elementSolutions])

	# First figure: dT/dQ and convergence metrics
	fig, (ax_top, ax_bottom) = plt.subplots(
		2, 1, figsize=(12, 9), sharex=True,
		gridspec_kw={"height_ratios": [2, 1]}
	)

	# Top subplot: dT and dQ with secondary axis
	ax_top_secondary = ax_top.twinx()
	line_dT, = ax_top.plot(radius, dT, marker='o', color='tab:blue', label='dT')
	line_dQ, = ax_top_secondary.plot(radius, dQ, marker='s', color='tab:orange', label='dQ')

	ax_top.set_ylabel('Differential Thrust dT (N/m)', color='tab:blue')
	ax_top_secondary.set_ylabel('Differential Torque dQ (Nm/m)', color='tab:orange')
	ax_top.tick_params(axis='y', labelcolor='tab:blue')
	ax_top_secondary.tick_params(axis='y', labelcolor='tab:orange')
	ax_top.grid(True)
	ax_top.legend([line_dT, line_dQ], ['dT', 'dQ'], loc='best')

	# Bottom subplot: convergence precision and iterations
	ax_bottom_secondary = ax_bottom.twinx()
	line_precision, = ax_bottom.plot(
		radius, convergencePrecision, marker='o', color='tab:green', label='Convergence precision'
	)
	line_iterations, = ax_bottom_secondary.plot(
		radius, convergenceIterations, marker='^', color='tab:red', label='Iterations'
	)

	ax_bottom.set_xlabel('Radius (m)')
	ax_bottom.set_ylabel('Convergence precision (-)', color='tab:green')
	ax_bottom_secondary.set_ylabel('Iterations (-)', color='tab:red')
	ax_bottom.tick_params(axis='y', labelcolor='tab:green')
	ax_bottom_secondary.tick_params(axis='y', labelcolor='tab:red')
	ax_bottom.grid(True)
	ax_bottom.legend([line_precision, line_iterations], ['Convergence precision', 'Iterations'], loc='best')

	plt.tight_layout()

	# Second figure: Prandtl correction and axial induction
	fig2 = plt.figure(figsize=(10, 6))
	fig2_ax = fig2.add_subplot(111)

	# Plot Prandtl correction
	ax2_main = fig2_ax
	ax2_main.plot(radius, prandtlCorrection, marker='o', color='tab:purple', label='Prandtl correction', linewidth=2)
	ax2_main.set_xlabel('Radius (m)')
	ax2_main.set_ylabel('Prandtl Tip Loss Correction (-)', color='tab:purple')
	ax2_main.tick_params(axis='y', labelcolor='tab:purple')
	ax2_main.grid(True)

	# Plot axial induction on secondary axis
	ax2_secondary = ax2_main.twinx()
	ax2_secondary.plot(radius, axialInduction, marker='s', color='tab:cyan', label='Axial induction a', linewidth=2)
	ax2_secondary.set_ylabel('Axial induction a (-)', color='tab:cyan')
	ax2_secondary.tick_params(axis='y', labelcolor='tab:cyan')

	ax2_main.set_title('Prandtl Tip Loss Correction vs Radius')
	ax2_main.legend(loc='upper left')
	ax2_secondary.legend(loc='upper right')
	plt.grid(True)

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
