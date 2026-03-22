import numpy as np
import matplotlib.pyplot as plt
import SolverTools as st
from pathlib import Path

plt.rcParams['figure.max_open_warning'] = 0

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
	- angle of attack distribution along the blade radial position
	- inflow angle distribution along the blade radial position
	- axial and azimuthal sectional force coefficients
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
	angleOfAttack = np.rad2deg(np.array([element.angleOfAttack for element in solution.elementSolutions]))
	inflowAngle = np.rad2deg(np.array([element.inflowAngle for element in solution.elementSolutions]))

	plt.figure()
	plt.plot(radialPosition, angleOfAttack, label='Angle of Attack (alpha)')
	plt.plot(radialPosition, inflowAngle, label='Inflow Angle (phi)')
	plt.xlabel('Radial Position (r/R)')
	plt.ylabel('Angle (degrees)')
	plt.legend()
	plt.grid(True)

	# Plotting the force coefficients
	cAxial = np.array([element.axialForceCoefficient for element in solution.elementSolutions])
	cAzimuthal = np.array([element.azimuthalForceCoefficient for element in solution.elementSolutions])

	plt.figure()
	plt.plot(radialPosition, cAxial, label='Axial Force Coefficient')
	plt.plot(radialPosition, cAzimuthal, label='Azimuthal Force Coefficient')
	plt.xlabel('Radial Position (r/R)')
	plt.ylabel('Coefficient')
	plt.legend()
	plt.grid(True)


def PlotSpanwiseResults(solution: st.SolverData):
	"""
	Create report-ready spanwise plots for one TSR case.
	Returns the created figure.
	"""
	tipRadius = solution.geometry.tipRadius
	radius = np.array([element.radius for element in solution.elementSolutions])
	radialPosition = radius / tipRadius

	inductionFactors = np.array([element.a for element in solution.elementSolutions])
	inductionFactorsPrime = np.array([element.aPrime for element in solution.elementSolutions])
	angleOfAttack = np.rad2deg(np.array([element.angleOfAttack for element in solution.elementSolutions]))
	inflowAngle = np.rad2deg(np.array([element.inflowAngle for element in solution.elementSolutions]))
	cAxial = np.array([element.axialForceCoefficient for element in solution.elementSolutions])
	cAzimuthal = np.array([element.azimuthalForceCoefficient for element in solution.elementSolutions])
	dT = solution.result.dT
	dQ = solution.result.dQ

	fig, axs = plt.subplots(2, 2, figsize=(13, 9))
	fig.suptitle(f'Spanwise Results for TSR = {solution.geometry.tipSpeedRatio:.1f}')

	axs[0, 0].plot(radialPosition, inductionFactors, label='a', linewidth=2)
	axs[0, 0].plot(radialPosition, inductionFactorsPrime, label="a'", linewidth=2)
	axs[0, 0].set_xlabel('Radial Position (r/R)')
	axs[0, 0].set_ylabel('Induction Factor [-]')
	axs[0, 0].grid(True)
	axs[0, 0].legend()

	axs[0, 1].plot(radialPosition, angleOfAttack, label='alpha', linewidth=2)
	axs[0, 1].plot(radialPosition, inflowAngle, label='phi', linewidth=2)
	axs[0, 1].set_xlabel('Radial Position (r/R)')
	axs[0, 1].set_ylabel('Angle [deg]')
	axs[0, 1].grid(True)
	axs[0, 1].legend()

	axs[1, 0].plot(radialPosition, cAxial, label='C axial', linewidth=2)
	axs[1, 0].plot(radialPosition, cAzimuthal, label='C azimuthal', linewidth=2)
	axs[1, 0].set_xlabel('Radial Position (r/R)')
	axs[1, 0].set_ylabel('Section Coefficient [-]')
	axs[1, 0].grid(True)
	axs[1, 0].legend()

	loadAxis = axs[1, 1]
	torqueAxis = loadAxis.twinx()
	loadLine = loadAxis.plot(radialPosition, dT, label='dT', linewidth=2, color='tab:blue')
	torqueLine = torqueAxis.plot(radialPosition, dQ, label='dQ', linewidth=2, color='tab:orange')
	loadAxis.set_xlabel('Radial Position (r/R)')
	loadAxis.set_ylabel('dT [N/m]', color='tab:blue')
	torqueAxis.set_ylabel('dQ [Nm/m]', color='tab:orange')
	loadAxis.tick_params(axis='y', labelcolor='tab:blue')
	torqueAxis.tick_params(axis='y', labelcolor='tab:orange')
	loadAxis.grid(True)
	loadAxis.legend(loadLine + torqueLine, ['dT', 'dQ'], loc='upper left')

	fig.tight_layout()
	return fig


def PlotPerformanceComparison(solutions: list[st.SolverData]):
	"""
	Create comparison plots across tip-speed ratios.
	Returns the created figure.
	"""
	tsr = np.array([solution.geometry.tipSpeedRatio for solution in solutions])
	thrust = np.array([solution.result.totalThrust for solution in solutions])
	torque = np.array([solution.result.totalTorque for solution in solutions])
	cp = np.array([solution.result.cP for solution in solutions])
	ct = np.array([
		solution.result.totalThrust / (
			0.5 * solution.airDensity * np.pi * solution.geometry.tipRadius ** 2 * solution.geometry.freeStreamVelocity ** 2
		)
		for solution in solutions
	])
	power = torque * np.array([solution.geometry.rotationalSpeed for solution in solutions])

	fig, axs = plt.subplots(3, 2, figsize=(12, 11))
	fig.suptitle('Baseline Rotor Performance Comparison')

	axs[0, 0].plot(tsr, cp, marker='o', linewidth=2)
	axs[0, 0].set_xlabel('Tip Speed Ratio [-]')
	axs[0, 0].set_ylabel('CP [-]')
	axs[0, 0].grid(True)

	axs[0, 1].plot(tsr, ct, marker='o', linewidth=2)
	axs[0, 1].set_xlabel('Tip Speed Ratio [-]')
	axs[0, 1].set_ylabel('CT [-]')
	axs[0, 1].grid(True)

	axs[1, 0].plot(tsr, thrust, marker='o', linewidth=2)
	axs[1, 0].set_xlabel('Tip Speed Ratio [-]')
	axs[1, 0].set_ylabel('Thrust [N]')
	axs[1, 0].grid(True)

	axs[1, 1].plot(tsr, torque, marker='o', linewidth=2)
	axs[1, 1].set_xlabel('Tip Speed Ratio [-]')
	axs[1, 1].set_ylabel('Torque [Nm]')
	axs[1, 1].grid(True)

	axs[2, 0].plot(tsr, power, marker='o', linewidth=2)
	axs[2, 0].set_xlabel('Tip Speed Ratio [-]')
	axs[2, 0].set_ylabel('Power [W]')
	axs[2, 0].grid(True)

	axs[2, 1].axis('off')

	fig.tight_layout()
	return fig


def PlotAnglesComparison(solutions: list[st.SolverData]):
	fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
	fig.suptitle('Task d.a: Spanwise Angle Distributions')
	for solution in solutions:
		radial = solution.result.radius / solution.geometry.tipRadius
		alpha = np.rad2deg(np.array([element.angleOfAttack for element in solution.elementSolutions]))
		phi = np.rad2deg(np.array([element.inflowAngle for element in solution.elementSolutions]))
		label = f"TSR = {solution.geometry.tipSpeedRatio:.0f}"
		axs[0].plot(radial, alpha, linewidth=2, label=label)
		axs[1].plot(radial, phi, linewidth=2, label=label)

	axs[0].set_xlabel('r/R [-]')
	axs[0].set_ylabel('alpha [deg]')
	axs[0].grid(True)
	axs[0].legend()

	axs[1].set_xlabel('r/R [-]')
	axs[1].set_ylabel('phi [deg]')
	axs[1].grid(True)
	axs[1].legend()

	fig.tight_layout()
	return fig


def PlotInductionComparison(solutions: list[st.SolverData]):
	fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
	fig.suptitle('Task d.b: Spanwise Induction Distributions')
	for solution in solutions:
		radial = solution.result.radius / solution.geometry.tipRadius
		a = np.array([element.a for element in solution.elementSolutions])
		aPrime = np.array([element.aPrime for element in solution.elementSolutions])
		label = f"TSR = {solution.geometry.tipSpeedRatio:.0f}"
		axs[0].plot(radial, a, linewidth=2, label=label)
		axs[1].plot(radial, aPrime, linewidth=2, label=label)

	axs[0].set_xlabel('r/R [-]')
	axs[0].set_ylabel('a [-]')
	axs[0].grid(True)
	axs[0].legend()

	axs[1].set_xlabel('r/R [-]')
	axs[1].set_ylabel("a' [-]")
	axs[1].grid(True)
	axs[1].legend()

	fig.tight_layout()
	return fig


def PlotLoadingComparison(solutions: list[st.SolverData]):
	fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
	fig.suptitle('Task d.c: Spanwise Coefficients Cn, Ct, Cq')
	for solution in solutions:
		radial = solution.result.radius / solution.geometry.tipRadius
		cn = np.array([element.axialForceCoefficient for element in solution.elementSolutions])
		ct = np.array([element.azimuthalForceCoefficient for element in solution.elementSolutions])
		cq = ct * radial
		label = f"TSR = {solution.geometry.tipSpeedRatio:.0f}"
		axs[0].plot(radial, cn, linewidth=2, label=label)
		axs[1].plot(radial, ct, linewidth=2, label=label)
		axs[2].plot(radial, cq, linewidth=2, label=label)

	axs[0].set_xlabel('r/R [-]')
	axs[0].set_ylabel('Cn [-]')
	axs[0].grid(True)
	axs[0].legend()

	axs[1].set_xlabel('r/R [-]')
	axs[1].set_ylabel('Ct [-]')
	axs[1].grid(True)
	axs[1].legend()

	axs[2].set_xlabel('r/R [-]')
	axs[2].set_ylabel('Cq [-]')
	axs[2].grid(True)
	axs[2].legend()

	fig.tight_layout()
	return fig


def PlotTipCorrectionTotals(withTipSolutions: list[st.SolverData], withoutTipSolutions: list[st.SolverData]):
	fig, axs = plt.subplots(2, 2, figsize=(12, 8))
	fig.suptitle('Task e: Influence of Tip/Root Correction on Integrated Performance')

	tsr = np.array([solution.geometry.tipSpeedRatio for solution in withTipSolutions])
	cp_with = np.array([solution.result.cP for solution in withTipSolutions])
	cp_without = np.array([solution.result.cP for solution in withoutTipSolutions])
	ct_with = np.array([
		solution.result.totalThrust / (
			0.5 * solution.airDensity * np.pi * solution.geometry.tipRadius ** 2 * solution.geometry.freeStreamVelocity ** 2
		)
		for solution in withTipSolutions
	])
	ct_without = np.array([
		solution.result.totalThrust / (
			0.5 * solution.airDensity * np.pi * solution.geometry.tipRadius ** 2 * solution.geometry.freeStreamVelocity ** 2
		)
		for solution in withoutTipSolutions
	])
	thrust_with = np.array([solution.result.totalThrust for solution in withTipSolutions])
	thrust_without = np.array([solution.result.totalThrust for solution in withoutTipSolutions])
	torque_with = np.array([solution.result.totalTorque for solution in withTipSolutions])
	torque_without = np.array([solution.result.totalTorque for solution in withoutTipSolutions])

	series = [
		(axs[0, 0], cp_with, cp_without, 'CP [-]'),
		(axs[0, 1], ct_with, ct_without, 'CT [-]'),
		(axs[1, 0], thrust_with, thrust_without, 'Thrust [N]'),
		(axs[1, 1], torque_with, torque_without, 'Torque [Nm]'),
	]
	for axis, with_values, without_values, ylabel in series:
		axis.plot(tsr, with_values, marker='o', linewidth=2, label='With Prandtl')
		axis.plot(tsr, without_values, marker='s', linewidth=2, label='Without Prandtl')
		axis.set_xlabel('Tip Speed Ratio [-]')
		axis.set_ylabel(ylabel)
		axis.grid(True)
		axis.legend()

	fig.tight_layout()
	return fig


def PlotTipCorrectionSpanwise(withTip: st.SolverData, withoutTip: st.SolverData):
	fig, axs = plt.subplots(2, 2, figsize=(12, 8))
	fig.suptitle(f'Task e: Spanwise Influence of Tip/Root Correction at TSR = {withTip.geometry.tipSpeedRatio:.0f}')

	radial = withTip.result.radius / withTip.geometry.tipRadius
	phi_with = np.rad2deg(np.array([element.inflowAngle for element in withTip.elementSolutions]))
	phi_without = np.rad2deg(np.array([element.inflowAngle for element in withoutTip.elementSolutions]))
	a_with = np.array([element.a for element in withTip.elementSolutions])
	a_without = np.array([element.a for element in withoutTip.elementSolutions])
	factor_with = np.array([element.prandtlCorrection for element in withTip.elementSolutions])

	axs[0, 0].plot(radial, factor_with, linewidth=2)
	axs[0, 0].set_xlabel('Radial Position (r/R)')
	axs[0, 0].set_ylabel('Prandtl Factor F [-]')
	axs[0, 0].grid(True)

	axs[0, 1].plot(radial, a_with, linewidth=2, label='With Prandtl')
	axs[0, 1].plot(radial, a_without, linewidth=2, label='Without Prandtl')
	axs[0, 1].set_xlabel('Radial Position (r/R)')
	axs[0, 1].set_ylabel('Axial Induction a [-]')
	axs[0, 1].grid(True)
	axs[0, 1].legend()

	axs[1, 0].plot(radial, withTip.result.dT, linewidth=2, label='With Prandtl')
	axs[1, 0].plot(radial, withoutTip.result.dT, linewidth=2, label='Without Prandtl')
	axs[1, 0].set_xlabel('Radial Position (r/R)')
	axs[1, 0].set_ylabel('Thrust Loading dT [N/m]')
	axs[1, 0].grid(True)
	axs[1, 0].legend()

	axs[1, 1].plot(radial, phi_with, linewidth=2, label='With Prandtl')
	axs[1, 1].plot(radial, phi_without, linewidth=2, label='Without Prandtl')
	axs[1, 1].set_xlabel('Radial Position (r/R)')
	axs[1, 1].set_ylabel('Inflow Angle [deg]')
	axs[1, 1].grid(True)
	axs[1, 1].legend()

	fig.tight_layout()
	return fig


def PlotAnnuliStudy(studyRows: list[dict], referenceThrust: float):
	fig, axs = plt.subplots(2, 2, figsize=(12, 8))
	fig.suptitle('Task f: Annuli Count, Spacing Method, and Total-Thrust Convergence')

	spacing_methods = sorted({row['spacing'] for row in studyRows})
	for spacing in spacing_methods:
		subset = [row for row in studyRows if row['spacing'] == spacing]
		counts = np.array([row['annuli'] for row in subset])
		thrust = np.array([row['thrust'] for row in subset])
		cp = np.array([row['cp'] for row in subset])
		error = np.array([row['relative_error_percent'] for row in subset])
		iterations = np.array([row['mean_iterations'] for row in subset])
		label = spacing.capitalize()
		axs[0, 0].plot(counts, thrust, marker='o', linewidth=2, label=label)
		axs[0, 1].plot(counts, error, marker='o', linewidth=2, label=label)
		axs[1, 0].plot(counts, cp, marker='o', linewidth=2, label=label)
		axs[1, 1].plot(counts, iterations, marker='o', linewidth=2, label=label)

	axs[0, 0].axhline(referenceThrust, color='k', linestyle='--', linewidth=1.5, label='Reference')
	axs[0, 0].set_xlabel('Number of Annuli [-]')
	axs[0, 0].set_ylabel('Total Thrust [N]')
	axs[0, 0].grid(True)
	axs[0, 0].legend()

	axs[0, 1].set_xlabel('Number of Annuli [-]')
	axs[0, 1].set_ylabel('Thrust Error [%]')
	axs[0, 1].grid(True)
	axs[0, 1].legend()

	axs[1, 0].set_xlabel('Number of Annuli [-]')
	axs[1, 0].set_ylabel('CP [-]')
	axs[1, 0].grid(True)
	axs[1, 0].legend()

	axs[1, 1].set_xlabel('Number of Annuli [-]')
	axs[1, 1].set_ylabel('Mean Iterations per Element [-]')
	axs[1, 1].grid(True)
	axs[1, 1].legend()

	fig.tight_layout()
	return fig


def PlotSpacingDistribution(constantEdges: np.ndarray, cosineEdges: np.ndarray):
	fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
	fig.suptitle('Task f: Radial Spacing Methods')

	constant_centers = 0.5 * (constantEdges[:-1] + constantEdges[1:])
	cosine_centers = 0.5 * (cosineEdges[:-1] + cosineEdges[1:])
	constant_width = np.diff(constantEdges)
	cosine_width = np.diff(cosineEdges)

	axs[0].plot(np.arange(1, len(constant_centers) + 1), constant_centers, marker='o', linewidth=2, label='Constant')
	axs[0].plot(np.arange(1, len(cosine_centers) + 1), cosine_centers, marker='s', linewidth=2, label='Cosine')
	axs[0].set_xlabel('Annulus Index [-]')
	axs[0].set_ylabel('Annulus Center r [m]')
	axs[0].grid(True)
	axs[0].legend()

	axs[1].plot(np.arange(1, len(constant_width) + 1), constant_width, marker='o', linewidth=2, label='Constant')
	axs[1].plot(np.arange(1, len(cosine_width) + 1), cosine_width, marker='s', linewidth=2, label='Cosine')
	axs[1].set_xlabel('Annulus Index [-]')
	axs[1].set_ylabel('Annulus Width [m]')
	axs[1].grid(True)
	axs[1].legend()

	fig.tight_layout()
	return fig


def PlotAirfoilOperationalPolar(alphaPolarDeg, clPolar, cdPolar, caseAlphaDeg, caseCl, caseCd, radialPosition):
	fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
	fig.suptitle('Task j: Airfoil Operational Points for Selected Baseline Case')

	scatter = axs[0].scatter(caseAlphaDeg, caseCl, c=radialPosition, cmap='viridis', s=25, label='Operating points')
	axs[0].plot(alphaPolarDeg, clPolar, linewidth=2, color='tab:blue', label='Polar')
	axs[0].set_xlabel('Angle of Attack [deg]')
	axs[0].set_ylabel('Lift Coefficient Cl [-]')
	axs[0].grid(True)
	axs[0].legend()

	axs[1].scatter(caseAlphaDeg, caseCd, c=radialPosition, cmap='viridis', s=25, label='Operating points')
	axs[1].plot(alphaPolarDeg, cdPolar, linewidth=2, color='tab:red', label='Polar')
	axs[1].set_xlabel('Angle of Attack [deg]')
	axs[1].set_ylabel('Drag Coefficient Cd [-]')
	axs[1].grid(True)
	axs[1].legend()

	colorbar = fig.colorbar(scatter, ax=axs, shrink=0.9)
	colorbar.set_label('Radial Position r/R [-]')
	return fig


def PlotAirfoilSpanwise(radialPosition, alphaDeg, cl, cd, clToCd, chord):
	fig, axs = plt.subplots(2, 2, figsize=(12, 8))
	fig.suptitle('Task j: Spanwise Airfoil Operating State and Chord Relation')

	axs[0, 0].plot(radialPosition, alphaDeg, linewidth=2)
	axs[0, 0].set_xlabel('Radial Position (r/R)')
	axs[0, 0].set_ylabel('Angle of Attack [deg]')
	axs[0, 0].grid(True)

	axs[0, 1].plot(radialPosition, cl, linewidth=2, label='Cl')
	axs[0, 1].plot(radialPosition, cd, linewidth=2, label='Cd')
	axs[0, 1].set_xlabel('Radial Position (r/R)')
	axs[0, 1].set_ylabel('Coefficient [-]')
	axs[0, 1].grid(True)
	axs[0, 1].legend()

	axs[1, 0].plot(radialPosition, clToCd, linewidth=2)
	axs[1, 0].set_xlabel('Radial Position (r/R)')
	axs[1, 0].set_ylabel('Cl/Cd [-]')
	axs[1, 0].grid(True)

	chordAxis = axs[1, 1]
	clAxis = chordAxis.twinx()
	chordLine = chordAxis.plot(radialPosition, chord, linewidth=2, color='tab:blue', label='Chord')
	clLine = clAxis.plot(radialPosition, cl, linewidth=2, color='tab:orange', label='Cl')
	chordAxis.set_xlabel('Radial Position (r/R)')
	chordAxis.set_ylabel('Chord [m]', color='tab:blue')
	clAxis.set_ylabel('Cl [-]', color='tab:orange')
	chordAxis.tick_params(axis='y', labelcolor='tab:blue')
	clAxis.tick_params(axis='y', labelcolor='tab:orange')
	chordAxis.grid(True)
	chordAxis.legend(chordLine + clLine, ['Chord', 'Cl'], loc='upper right')

	fig.tight_layout()
	return fig


def _DesignSpanwiseData(evaluation):
	solution = evaluation.solution
	radial = solution.result.radius / solution.geometry.tipRadius
	alpha = np.rad2deg(np.array([element.angleOfAttack for element in solution.elementSolutions]))
	phi = np.rad2deg(np.array([element.inflowAngle for element in solution.elementSolutions]))
	a = np.array([element.a for element in solution.elementSolutions])
	aPrime = np.array([element.aPrime for element in solution.elementSolutions])
	chord = np.array([solution.geometry.chordFunction(radius) for radius in solution.result.radius])
	twist = np.array([solution.geometry.bladeTwist(radius) for radius in solution.result.radius])
	return {
		"radial": radial,
		"alpha": alpha,
		"phi": phi,
		"a": a,
		"a_prime": aPrime,
		"dT": solution.result.dT,
		"dQ": solution.result.dQ,
		"chord": chord,
		"twist": twist,
	}


def PlotDesignSearchVsTwist(stageBResults):
	ordered = sorted(stageBResults, key=lambda evaluation: evaluation.twist_scale)
	scales = np.array([evaluation.twist_scale for evaluation in ordered])
	cp = np.array([evaluation.cp for evaluation in ordered])
	pitch = np.array([evaluation.pitch_deg for evaluation in ordered])

	fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
	fig.suptitle('Point 2 Stage B: Twist-Scale Search at CT = 0.75')

	axs[0].plot(scales, cp, linewidth=2)
	axs[0].set_xlabel('Twist Scale [-]')
	axs[0].set_ylabel('CP [-]')
	axs[0].grid(True)

	axs[1].plot(scales, pitch, linewidth=2)
	axs[1].set_xlabel('Twist Scale [-]')
	axs[1].set_ylabel('Required Pitch [deg]')
	axs[1].grid(True)

	fig.tight_layout()
	return fig


def PlotDesignSearchVsChord(stageCResults):
	ordered = sorted(stageCResults, key=lambda evaluation: evaluation.chord_scale)
	scales = np.array([evaluation.chord_scale for evaluation in ordered])
	cp = np.array([evaluation.cp for evaluation in ordered])
	pitch = np.array([evaluation.pitch_deg for evaluation in ordered])

	fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
	fig.suptitle('Point 2 Stage C: Chord-Scale Search at CT = 0.75')

	axs[0].plot(scales, cp, linewidth=2)
	axs[0].set_xlabel('Chord Scale [-]')
	axs[0].set_ylabel('CP [-]')
	axs[0].grid(True)

	axs[1].plot(scales, pitch, linewidth=2)
	axs[1].set_xlabel('Chord Scale [-]')
	axs[1].set_ylabel('Required Pitch [deg]')
	axs[1].grid(True)

	fig.tight_layout()
	return fig


def PlotDesignComparison(baseline, stageA, stageB, stageC):
	fig, axs = plt.subplots(2, 2, figsize=(12, 8))
	fig.suptitle('Point 2: Baseline vs Stage A/B/C at TSR = 8')

	evaluations = [baseline, stageA, stageB, stageC]
	labels = ['Baseline', 'Stage A', 'Stage B', 'Stage C']

	for evaluation, label in zip(evaluations, labels):
		data = _DesignSpanwiseData(evaluation)
		axs[0, 0].plot(data["radial"], data["alpha"], linewidth=2, label=label)
		axs[0, 1].plot(data["radial"], data["a"], linewidth=2, label=label)
		axs[1, 0].plot(data["radial"], data["dT"], linewidth=2, label=label)
		axs[1, 1].plot(data["radial"], data["dQ"], linewidth=2, label=label)

	axs[0, 0].set_xlabel('Radial Position (r/R)')
	axs[0, 0].set_ylabel('Angle of Attack [deg]')
	axs[0, 0].grid(True)
	axs[0, 0].legend()

	axs[0, 1].set_xlabel('Radial Position (r/R)')
	axs[0, 1].set_ylabel('Axial Induction a [-]')
	axs[0, 1].grid(True)
	axs[0, 1].legend()

	axs[1, 0].set_xlabel('Radial Position (r/R)')
	axs[1, 0].set_ylabel('Thrust Loading dT [N/m]')
	axs[1, 0].grid(True)
	axs[1, 0].legend()

	axs[1, 1].set_xlabel('Radial Position (r/R)')
	axs[1, 1].set_ylabel('Azimuthal Loading dQ [Nm/m]')
	axs[1, 1].grid(True)
	axs[1, 1].legend()

	fig.tight_layout()
	return fig


def PlotDesignDistributions(baseline, stageA, stageB, stageC):
	fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
	fig.suptitle('Point 2: Blade-Design Distributions')

	evaluations = [baseline, stageA, stageB, stageC]
	labels = ['Baseline', 'Stage A', 'Stage B', 'Stage C']
	for evaluation, label in zip(evaluations, labels):
		data = _DesignSpanwiseData(evaluation)
		axs[0].plot(data["radial"], data["twist"], linewidth=2, label=label)
		axs[1].plot(data["radial"], data["chord"], linewidth=2, label=label)

	axs[0].set_xlabel('Radial Position (r/R)')
	axs[0].set_ylabel('Twist [deg]')
	axs[0].grid(True)
	axs[0].legend()

	axs[1].set_xlabel('Radial Position (r/R)')
	axs[1].set_ylabel('Chord [m]')
	axs[1].grid(True)
	axs[1].legend()

	fig.tight_layout()
	return fig


def PlotDesignSummaryTable(baseline, stageA, stageB, stageC, targetCt: float, actuatorCp: float):
	rows = [
		['Baseline', f'{baseline.pitch_deg:.3f}', f'{baseline.twist_scale:.3f}', f'{baseline.chord_scale:.3f}', f'{baseline.ct:.4f}', f'{baseline.cp:.4f}'],
		['Stage A', f'{stageA.pitch_deg:.3f}', f'{stageA.twist_scale:.3f}', f'{stageA.chord_scale:.3f}', f'{stageA.ct:.4f}', f'{stageA.cp:.4f}'],
		['Stage B', f'{stageB.pitch_deg:.3f}', f'{stageB.twist_scale:.3f}', f'{stageB.chord_scale:.3f}', f'{stageB.ct:.4f}', f'{stageB.cp:.4f}'],
		['Stage C', f'{stageC.pitch_deg:.3f}', f'{stageC.twist_scale:.3f}', f'{stageC.chord_scale:.3f}', f'{stageC.ct:.4f}', f'{stageC.cp:.4f}'],
		['Actuator disk', '-', '-', '-', f'{targetCt:.4f}', f'{actuatorCp:.4f}'],
	]

	fig, ax = plt.subplots(figsize=(10, 3.2))
	ax.axis('off')
	table = ax.table(
		cellText=rows,
		colLabels=['Case', 'Pitch [deg]', 'Twist Scale [-]', 'Chord Scale [-]', 'CT [-]', 'CP [-]'],
		loc='center')
	table.auto_set_font_size(False)
	table.set_fontsize(10)
	table.scale(1, 1.5)
	fig.tight_layout()
	return fig


def PlotFinalDesignPerformance(baseline, redesign):
	baselineData = _DesignSpanwiseData(baseline)
	redesignData = _DesignSpanwiseData(redesign)

	fig, axs = plt.subplots(3, 2, figsize=(12, 10))
	fig.suptitle('Task h: Baseline vs Final Redesign Performance at TSR = 8')

	series = [
		(axs[0, 0], "alpha", 'Angle of Attack [deg]'),
		(axs[0, 1], "phi", 'Inflow Angle [deg]'),
		(axs[1, 0], "a", 'Axial Induction a [-]'),
		(axs[1, 1], "a_prime", "Azimuthal Induction a' [-]"),
		(axs[2, 0], "dT", 'Thrust Loading dT [N/m]'),
		(axs[2, 1], "dQ", 'Azimuthal Loading dQ [Nm/m]'),
	]

	for axis, key, ylabel in series:
		axis.plot(baselineData["radial"], baselineData[key], linewidth=2, label='Baseline')
		axis.plot(redesignData["radial"], redesignData[key], linewidth=2, label='Final redesign')
		axis.set_xlabel('Radial Position (r/R)')
		axis.set_ylabel(ylabel)
		axis.grid(True)
		axis.legend()

	fig.tight_layout()
	return fig


def PlotFinalDesignGeometry(baseline, redesign):
	baselineData = _DesignSpanwiseData(baseline)
	redesignData = _DesignSpanwiseData(redesign)

	fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
	fig.suptitle('Task h: Baseline vs Final Redesign Geometry')

	axs[0].plot(baselineData["radial"], baselineData["twist"], linewidth=2, label='Baseline')
	axs[0].plot(redesignData["radial"], redesignData["twist"], linewidth=2, label='Final redesign')
	axs[0].set_xlabel('Radial Position (r/R)')
	axs[0].set_ylabel('Twist [deg]')
	axs[0].grid(True)
	axs[0].legend()

	axs[1].plot(baselineData["radial"], baselineData["chord"], linewidth=2, label='Baseline')
	axs[1].plot(redesignData["radial"], redesignData["chord"], linewidth=2, label='Final redesign')
	axs[1].set_xlabel('Radial Position (r/R)')
	axs[1].set_ylabel('Chord [m]')
	axs[1].grid(True)
	axs[1].legend()

	fig.tight_layout()
	return fig


def PlotFinalDesignOperatingPoints(baseline, redesign):
	airfoil = redesign.solution.geometry.airfoil
	alphaPolarDeg = np.rad2deg(airfoil.alpha)

	baselineData = _DesignSpanwiseData(baseline)
	redesignData = _DesignSpanwiseData(redesign)

	baselineAlphaRad = np.deg2rad(baselineData["alpha"])
	redesignAlphaRad = np.deg2rad(redesignData["alpha"])
	baselineCl = np.array([airfoil.lookup(alpha, 'cl') for alpha in baselineAlphaRad])
	baselineCd = np.array([airfoil.lookup(alpha, 'cd') for alpha in baselineAlphaRad])
	redesignCl = np.array([airfoil.lookup(alpha, 'cl') for alpha in redesignAlphaRad])
	redesignCd = np.array([airfoil.lookup(alpha, 'cd') for alpha in redesignAlphaRad])

	fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
	fig.suptitle('Task h: Airfoil Operating Points for Baseline and Final Redesign')

	axs[0].plot(alphaPolarDeg, airfoil.cl, linewidth=2, color='tab:blue', label='Polar')
	axs[0].scatter(baselineData["alpha"], baselineCl, s=20, label='Baseline')
	axs[0].scatter(redesignData["alpha"], redesignCl, s=20, label='Final redesign')
	axs[0].set_xlabel('Angle of Attack [deg]')
	axs[0].set_ylabel('Lift Coefficient Cl [-]')
	axs[0].grid(True)
	axs[0].legend()

	axs[1].plot(alphaPolarDeg, airfoil.cd, linewidth=2, color='tab:red', label='Polar')
	axs[1].scatter(baselineData["alpha"], baselineCd, s=20, label='Baseline')
	axs[1].scatter(redesignData["alpha"], redesignCd, s=20, label='Final redesign')
	axs[1].set_xlabel('Angle of Attack [deg]')
	axs[1].set_ylabel('Drag Coefficient Cd [-]')
	axs[1].grid(True)
	axs[1].legend()

	fig.tight_layout()
	return fig


def PlotStagnationPressureDistribution(baselinePressure: dict, redesignPressure: dict):
	fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
	fig.suptitle('Task i: Stagnation Pressure Distribution for the Final Redesign')

	axs[0].plot(redesignPressure["radial"], redesignPressure["p0_upstream_star"], linewidth=2, label='1: Far upstream')
	axs[0].plot(redesignPressure["radial"], redesignPressure["p0_before_disk_star"], linewidth=2, linestyle='--', label='2: Before disk')
	axs[0].plot(redesignPressure["radial"], redesignPressure["p0_after_disk_star"], linewidth=2, label='3: After disk')
	axs[0].plot(redesignPressure["radial"], redesignPressure["p0_far_wake_star"], linewidth=2, linestyle='--', label='4: Far wake')
	axs[0].set_xlabel('Radial Position (r/R)')
	axs[0].set_ylabel('Normalized Stagnation Pressure [-]')
	axs[0].grid(True)
	axs[0].legend()

	axs[1].plot(baselinePressure["radial"], baselinePressure["p0_after_disk_star"], linewidth=2, label='Baseline')
	axs[1].plot(redesignPressure["radial"], redesignPressure["p0_after_disk_star"], linewidth=2, label='Final redesign')
	axs[1].set_xlabel('Radial Position (r/R)')
	axs[1].set_ylabel('Wake Stagnation Pressure [-]')
	axs[1].grid(True)
	axs[1].legend()

	fig.tight_layout()
	return fig


def SaveFigure(fig, outputPath: str):
	"""
	Save a matplotlib figure to disk.
	"""
	if fig._suptitle is not None:
		fig._suptitle.set_text('')
	for axis in fig.axes:
		axis.set_title('')

	path = Path(outputPath)
	path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(path, dpi=200, bbox_inches='tight')
