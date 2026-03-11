import numpy as np
import matplotlib.pyplot as plt
import SolverTools as st

# ---------------------------------------------------------------------------
# Shared style helper
# ---------------------------------------------------------------------------

def ApplyStyle(ax):
	"""Apply a consistent visual style to a matplotlib Axes object.

	Call this after adding all artists to an axes to enforce uniform
	font sizes, line weights, label colours, and grid appearance.
	"""
	FONT_SIZE   = 11
	LABEL_COLOR = "#222222"
	GRID_COLOR  = "#cccccc"

	# Tick and label font sizes
	ax.tick_params(labelsize=FONT_SIZE - 1, colors=LABEL_COLOR)
	ax.xaxis.label.set_size(FONT_SIZE)
	ax.yaxis.label.set_size(FONT_SIZE)
	ax.xaxis.label.set_color(LABEL_COLOR)
	ax.yaxis.label.set_color(LABEL_COLOR)

	# Title
	ax.title.set_size(FONT_SIZE + 1)
	ax.title.set_color(LABEL_COLOR)

	# Legend
	ax.legend()
	legend = ax.get_legend()

	# Thicken all plotted lines
	for line in ax.get_lines():
		line.set_linewidth(2.0)

	# Spine (border) color and weight
	for spine in ax.spines.values():
		spine.set_edgecolor(LABEL_COLOR)
		spine.set_linewidth(1.0)

	# Grid style
	ax.grid(True, color=GRID_COLOR, linewidth=0.8, linestyle="--")
	ax.set_axisbelow(True)

# ---------------------------------------------------------------------------

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
	ApplyStyle(axs[0, 0])

	# Iteration count plot
	axs[0, 1].plot(radius, convergenceIterations, marker='o', color='orange')
	axs[0, 1].set_title('Iteration Count vs Radius')
	axs[0, 1].set_xlabel('Radius (m)')
	axs[0, 1].set_ylabel('Iteration Count')
	ApplyStyle(axs[0, 1])

	# Prandtl correction plot
	axs[1, 0].plot(radius, prandtlCorrection, marker='o', color='green')
	axs[1, 0].set_title('Prandtl Correction vs Radius')
	axs[1, 0].set_xlabel('Radius (m)')
	axs[1, 0].set_ylabel('Prandtl Correction')
	ApplyStyle(axs[1, 0])

	# Axial induction factor plot
	axs[1, 1].plot(radius, axialInduction, marker='o', color='red')
	axs[1, 1].set_title('Axial Induction Factor (a) vs Radius')
	axs[1, 1].set_xlabel('Radius (m)')
	axs[1, 1].set_ylabel('Axial Induction Factor (a)')
	ApplyStyle(axs[1, 1])

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
	ApplyStyle(ax_surface)
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


	fig1, ax1 = plt.subplots()
	ax1.plot(radialPosition, inductionFactors, label='Axial Induction Factor (a)')
	ax1.plot(radialPosition, inductionFactorsPrime, label="Tangential Induction Factor (a')")
	ax1.set_xlabel('Radial Position (r/R)')
	ax1.set_ylabel('Induction Factor')
	ApplyStyle(ax1)

	# Plotting the angle of attack and inflow angle
	angleOfAttack = np.array([element.angleOfAttack for element in solution.elementSolutions])
	inflowAngle = np.array([element.inflowAngle for element in solution.elementSolutions])

	fig2, ax2 = plt.subplots()
	ax2.plot(radialPosition, np.rad2deg(angleOfAttack), label='Angle of Attack (alpha)')
	ax2.plot(radialPosition, np.rad2deg(inflowAngle), label='Inflow Angle (phi)')
	ax2.set_xlabel('Radial Position (r/R)')
	ax2.set_ylabel('Angle (degrees)')
	ApplyStyle(ax2)

	# Plotting the force coefficients
	Ct = np.array([element.dCt for element in solution.elementSolutions])
	Cn = np.array([element.dCn for element in solution.elementSolutions])
	Cq = np.array([element.dCq for element in solution.elementSolutions])

	fig3, ax3 = plt.subplots()
	ax3.plot(radialPosition, Ct, label='Thrust Coefficient (dCt)')
	ax3.plot(radialPosition, Cn, label='Normal Force Coefficient (dCn)')
	ax3.plot(radialPosition, Cq, label='Torque Coefficient (dCq)')
	ax3.set_xlabel('Radial Position (r/R)')
	ax3.set_ylabel('Coefficient')
	ApplyStyle(ax3)

##### Task e #####
def TipCorrectionPlot(solution: st.SolverData):
	"""
	Plot the Prandtl tip correction factor along the blade radial position.
	"""

	tipRadius = solution.geometry.tipRadius
	radius = np.array([element.radius for element in solution.elementSolutions])
	radialPosition = radius / tipRadius

	prandtlCorrection = np.array([element.prandtlCorrection for element in solution.elementSolutions])

	fig, ax = plt.subplots()
	ax.plot(radialPosition, prandtlCorrection, label='Prandtl Tip Correction Factor')
	ax.set_xlabel('Radial Position (r/R)')
	ax.set_ylabel('Prandtl Tip Correction Factor')
	ApplyStyle(ax)

def SpacingSensitivityPlot(solutionsList):
	"""
	Plot the convergence precision and iteration count for different element counts to analyze sensitivity to blade discretization.
	"""
	elementCounts = np.array([solution.elementCount for solution in solutionsList])
	convergencePrecisions = np.array([solution.getConvergencePrecision() for solution in solutionsList])
	iterationCounts = np.array([solution.getIterationsCount() for solution in solutionsList])

	fig, ax1 = plt.subplots()

	color = 'tab:blue'
	ax1.set_xlabel('Element Count')
	ax1.set_ylabel('Convergence Precision', color=color)
	ax1.plot(elementCounts, convergencePrecisions, marker='o', color=color)
	ax1.set_yscale('log')
	ax1.tick_params(axis='y', labelcolor=color)

	ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

	color = 'tab:orange'
	ax2.set_ylabel('Iteration Count', color=color)  # we already handled the x-label with ax1
	ax2.plot(elementCounts, iterationCounts, marker='o', color=color)
	ax2.tick_params(axis='y', labelcolor=color)

	fig.tight_layout()  # otherwise the right y-label is slightly clipped
	ApplyStyle(ax1)