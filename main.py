"""
Main Execution Script for BEM Solver
"""

import SolverTools as st
import Plotting
import numpy as np
import BEM

TSRs = np.array([6.0, 8.0, 10.0])

def plot_pitch_variations():	# Deprecated
	# Pitch as a function of radius
	localSolution = st.SolverData()
	localSolution.setParameters(
		maxIterations=1000,
		tolerance=1e-4,
		relaxation=0.5,
		elementCount=1)

	geometry = localSolution.geometry

	# Discretized map where each cell is one (radius, pitch) region
	pitchMinDeg = -10.0
	pitchMaxDeg = 10.0
	pitchBins = 50
	radiusBins = 80

	pitchGrid = np.linspace(pitchMinDeg, pitchMaxDeg, num=pitchBins)
	radiusGrid = np.linspace(geometry.bladeStart * geometry.tipRadius, geometry.bladeEnd * geometry.tipRadius - 0.001 * geometry.tipRadius, num=radiusBins)

	dTMap = np.zeros((pitchBins, radiusBins))
	dQMap = np.zeros((pitchBins, radiusBins))
	cPMap = np.zeros((pitchBins, radiusBins))

	for pitchIndex, pitchValue in enumerate(pitchGrid):
		for radiusIndex, radiusValue in enumerate(radiusGrid):
			

			# Override the blade pitch for this solution
			localSolution.geometry.bladePitch = pitchValue

			# NOTE: Pitch angle should be constant along the blade???
			# NOTE: There is an optimal inflow angle for maximum power, see cp of the airfoil
			#	Maybe we just need to enforce the optimal angle of attack for all radii to get best performance?
			# 	Could be compared to this approach I guess 

			dTValue, dQValue = BEM.solveElement(radiusValue, localSolution)
			dTMap[pitchIndex, radiusIndex] = dTValue
			dQMap[pitchIndex, radiusIndex] = dQValue
			
			# Calculate local cP
			localPower = dQValue * localSolution.geometry.rotationalSpeed
			localRotorArea = np.pi * localSolution.geometry.tipRadius**2
			localWindPower = 0.5 * localSolution.airDensity * localRotorArea * localSolution.geometry.freeStreamVelocity**3
			cPMap[pitchIndex, radiusIndex] = localPower / localWindPower if localWindPower > 0 else 0

	Plotting.plot_pitch_variations(radiusGrid, pitchGrid, cPMap)

def PartD():
	solutions = []
	for tipSpeedRatio in TSRs:
		solution = st.SolverData()
		solution.setParameters(
			maxIterations=500,
			tolerance=1e-6,
			relaxation=0.5,
			elementCount=100
		)
		solution.geometry.tipSpeedRatio = tipSpeedRatio
		BEM.SolveBEM(solution)
		solutions.append(solution)
	Plotting.PlotResults(solutions)

def PartF():
	# Prepare solutions for various numbers of annuli:
	solutionAnnuli = []
	minAnnuli = 5
	maxAnnuli = 500
	numOfAnnuli = 20
	annuliCounts = np.linspace(minAnnuli, maxAnnuli, num=numOfAnnuli, dtype=int)
	for annuliCount in annuliCounts:
		solution = st.SolverData()
		solution.setParameters(
			maxIterations=500,
			tolerance=1e-6,
			relaxation=0.5,
			elementCount=annuliCount
		)
		solution.geometry.tipSpeedRatio = 8.0
		BEM.SolveBEM(solution)
		solutionAnnuli.append(solution)

	Plotting.SpacingSensitivityPlot(solutionAnnuli)

if __name__ == "__main__":
	
	#### Part d ####
	PartD()

	#### Part f ####
	# PartF()

	Plotting.ShowPlots()