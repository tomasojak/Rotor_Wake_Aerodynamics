# Assignment 1; AE4135 Rotor/wake aerodynamics
# BEM Assignment
#
# Authors: - Tomas Jakus
#          - Codrin Chilea
#          - Sava Zamfirescu
#

# This file contains the wind-turbine BEM implementation used for the
# assignment. The update logic follows the course-slide formulation:
# blade-element loads are converted to annulus thrust, axial induction is
# updated through the momentum/Glauert relation, and Prandtl tip/root losses
# are applied to both induction factors.

import numpy as np
import SolverTools as st
EPSILON = 1e-6
DEBUG_PRINT = False


def BuildAnnulusEdges(solutionProperties: st.SolverData) -> np.ndarray:
    """
    Build annulus edges using the requested spacing method.
    """
    geometry = solutionProperties.geometry
    rootRadius = geometry.bladeStart * geometry.tipRadius
    tipRadius = (geometry.bladeEnd - solutionProperties.tipMarginFraction) * geometry.tipRadius
    elementCount = solutionProperties.elementCount

    if solutionProperties.spacingMethod == "cosine":
        theta = np.linspace(0.0, np.pi, elementCount + 1)
        distribution = 0.5 * (1 - np.cos(theta))
        return rootRadius + (tipRadius - rootRadius) * distribution

    return np.linspace(rootRadius, tipRadius, elementCount + 1)


def PitchAngle(radius, solverProperties: st.SolverData):
    bladeTwist = np.deg2rad(solverProperties.geometry.bladeTwist(radius))
    bladePitchDeg = np.deg2rad(solverProperties.geometry.bladePitch)
    return bladeTwist + bladePitchDeg  # rad


def PrandtlCorrection(radius, axialInduction, solverProperties: st.SolverData):
    """
	Calculate the combined Prandtl tip/root loss factor.

	Args:
		radius: Radial position along the blade (m)
		axialInduction: Axial induction factor used in the slide expression
		solverProperties: SolverData instance containing geometry and solver settings

    Returns:
		Combined Prandtl correction factor F (0 to 1)
	"""
    if not solverProperties.usePrandtlCorrection:
        return 1.0

    geometry = solverProperties.geometry

    r_R = geometry.dimensionlessRadialPosition(radius)
    root_R = geometry.bladeStart
    denominator = np.maximum((1 - axialInduction) ** 2, EPSILON)
    radial_term = np.sqrt(1 + (geometry.tipSpeedRatio * r_R) ** 2 / denominator)

    tip_argument = -geometry.bladeNumber / 2 * (geometry.bladeEnd - r_R) / r_R * radial_term
    root_argument = geometry.bladeNumber / 2 * (root_R - r_R) / r_R * radial_term

    f_tip = 2 / np.pi * np.arccos(np.clip(np.exp(np.clip(tip_argument, -700, 0)), 0.0, 1.0))
    f_root = 2 / np.pi * np.arccos(np.clip(np.exp(np.clip(root_argument, -700, 0)), 0.0, 1.0))

    if np.isnan(f_tip):
        f_tip = 0.0
    if np.isnan(f_root):
        f_root = 0.0

    return max(f_tip * f_root, 1e-3)


def GlauertCorrection(thrustCoefficient):
    """
	Convert annulus thrust coefficient into axial induction factor.

	Args:
		thrustCoefficient: Annulus thrust coefficient CT

	Returns:
		Axial induction factor a
	"""
    ct1 = 1.816
    ct2 = 2 * np.sqrt(ct1) - ct1

    if thrustCoefficient < ct2:
        return 0.5 - 0.5 * np.sqrt(1 - thrustCoefficient)

    return 1 + (thrustCoefficient - ct1) / (4 * np.sqrt(ct1) - 4)


def Inflowangle(radius, a, aPrime, solverProperties: st.SolverData):
    freeStreamVelocity = solverProperties.geometry.freeStreamVelocity
    rotationalSpeed = solverProperties.geometry.rotationalSpeed
    axialVelocity = freeStreamVelocity * (1 - a)
    tangentialVelocity = rotationalSpeed * radius * (1 + aPrime)
    return np.arctan2(axialVelocity, tangentialVelocity)  # rad


def AngleOfAttack(radius, a, aPrime, solverProperties: st.SolverData):
    return Inflowangle(radius, a, aPrime, solverProperties) - PitchAngle(radius, solverProperties)  # rad


def Solidity(radius, solverProperties: st.SolverData):
    bladeNumber = solverProperties.geometry.bladeNumber
    chord = solverProperties.geometry.chordFunction(radius)
    return bladeNumber * chord / (2 * np.pi * radius)


def BEMElement(radius, a, aPrime, solverProperties: st.SolverData) -> tuple:
    """
	Calculate sectional axial and azimuthal force coefficients.

	Args:
		radius: Radial position along the blade (m)
		a: Axial induction factor
		aPrime: Tangential induction factor
		solverProperties: SolverData instance containing geometry and solver settings

	Returns:
		Tuple of (cAxial, cAzimuthal)
	"""
    alpha = AngleOfAttack(radius, a, aPrime, solverProperties)
    cl, cd, _ = solverProperties.geometry.airfoil.lookupAll(alpha)
    inflowAngle = Inflowangle(radius, a, aPrime, solverProperties)

    cAxial = cl * np.cos(inflowAngle) + cd * np.sin(inflowAngle)
    cAzimuthal = cl * np.sin(inflowAngle) - cd * np.cos(inflowAngle)

    return cAxial, cAzimuthal


def BladeElementLoads(radius, a, aPrime, solverProperties: st.SolverData) -> tuple:
    """
    Compute 2D sectional loads from the sectional force coefficients.
    """
    geometry = solverProperties.geometry
    airDensity = solverProperties.airDensity
    cAxial, cAzimuthal = BEMElement(radius, a, aPrime, solverProperties)

    axialVelocity = geometry.freeStreamVelocity * (1 - a)
    tangentialVelocity = geometry.rotationalSpeed * radius * (1 + aPrime)
    relativeVelocity = np.sqrt(axialVelocity ** 2 + tangentialVelocity ** 2)

    dynamicPressure = 0.5 * airDensity * relativeVelocity ** 2
    chord = geometry.chordFunction(radius)

    axialLoad = dynamicPressure * chord * cAxial
    azimuthalLoad = dynamicPressure * chord * cAzimuthal

    return axialLoad, azimuthalLoad, cAxial, cAzimuthal

def SolveInduction(
        radius,
        annulusWidth,
        solutionProperties: st.SolverData) -> tuple:
    """
	Iteratively solve for induction factors at a given radial position.

	Args:
		radius: Radial position along the blade (m)
		annulusWidth: Width of the current annulus (m)
		solutionProperties: SolverData instance containing geometry, solver settings,
			and convergence history

	Returns:
		Tuple of (a, aPrime) - converged axial and tangential induction factors
	"""

    maxIterations = solutionProperties.maxIterations
    tolerance = solutionProperties.tolerance
    relaxation = solutionProperties.relaxation

    a = solutionProperties.initialGuess["a"]
    aPrime = solutionProperties.initialGuess["aPrime"]

    if solutionProperties.elementSolutions.__len__() > 0:
        if solutionProperties.elementSolutions[-1].converged:
            previousSolution = solutionProperties.elementSolutions[-1]
            a = previousSolution.a
            aPrime = previousSolution.aPrime

    geometry = solutionProperties.geometry
    airDensity = solutionProperties.airDensity
    annulusArea = 2 * np.pi * radius * annulusWidth

    for iteration in range(maxIterations):

        inflowAngle = Inflowangle(radius, a, aPrime, solutionProperties)
        angleOfAttack = AngleOfAttack(radius, a, aPrime, solutionProperties)
        axialLoad, azimuthalLoad, cAxial, cAzimuthal = BladeElementLoads(
            radius, a, aPrime, solutionProperties)

        thrust = axialLoad * geometry.bladeNumber * annulusWidth
        thrustCoefficient = thrust / (0.5 * airDensity * geometry.freeStreamVelocity ** 2 * annulusArea)

        aMomentum = GlauertCorrection(thrustCoefficient)
        prandtlCorrection = PrandtlCorrection(radius, aMomentum, solutionProperties)
        aMomentum /= prandtlCorrection
        if not np.isfinite(aMomentum):
            aMomentum = a
        aMomentum = float(np.clip(aMomentum, 0.0, 0.95))

        relaxedA = relaxation * aMomentum + (1 - relaxation) * a
        axialVelocity = np.maximum(geometry.freeStreamVelocity * (1 - relaxedA), EPSILON)
        denominator = 2 * airDensity * (2 * np.pi * radius) * axialVelocity * geometry.rotationalSpeed * radius ** 2
        aPrimeMomentum = azimuthalLoad * geometry.bladeNumber / np.maximum(denominator, EPSILON)
        aPrimeMomentum /= prandtlCorrection
        if not np.isfinite(aPrimeMomentum):
            aPrimeMomentum = aPrime
        aPrimeMomentum = float(np.clip(aPrimeMomentum, -0.25, 0.5))

        dCq = cAzimuthal * radius

        if np.abs(a - aMomentum) < tolerance and np.abs(aPrime - aPrimeMomentum) < tolerance:
            solution = st.IterationSolution(
                radius,
                aMomentum,
                aPrimeMomentum,
                iteration,
                max(np.abs(a - aMomentum),
                    np.abs(aPrime - aPrimeMomentum)),
                tolerance,
                prandtlCorrection,
                angleOfAttack,
                inflowAngle,
                cAxial,
                cAzimuthal,
                dCq)

            solutionProperties.addSolutionIteration(solution)

            return aMomentum, aPrimeMomentum

        a = relaxedA
        aPrime = relaxation * aPrimeMomentum + (1 - relaxation) * aPrime

    finalInflowAngle = Inflowangle(radius, a, aPrime, solutionProperties)
    finalAngleOfAttack = AngleOfAttack(radius, a, aPrime, solutionProperties)
    _, _, finalCAxial, finalCAzimuthal = BladeElementLoads(radius, a, aPrime, solutionProperties)
    finalPrandtlCorrection = PrandtlCorrection(radius, a, solutionProperties)
    finalDCq = finalCAzimuthal * radius

    solution = st.IterationSolution(
        radius,
        a,
        aPrime,
        maxIterations,
        max(np.abs(a - aMomentum), np.abs(aPrime - aPrimeMomentum)),
        tolerance,
        finalPrandtlCorrection,
        finalAngleOfAttack,
        finalInflowAngle,
        finalCAxial,
        finalCAzimuthal,
        finalDCq)

    solutionProperties.addSolutionIteration(solution)

    if DEBUG_PRINT:
        print("No convergence reached after maximum iterations")
    return a, aPrime


def solveElement(radius: float, annulusWidth: float, solutionProperties: st.SolverData = None) -> tuple:
    """
    Solve one annulus and return blade-integrated sectional thrust and torque
    per unit radial length.
    """
    a, aPrime = SolveInduction(radius, annulusWidth, solutionProperties)
    axialLoad, azimuthalLoad, _, _ = BladeElementLoads(radius, a, aPrime, solutionProperties)

    geometry = solutionProperties.geometry
    dT = axialLoad * geometry.bladeNumber
    dQ = azimuthalLoad * radius * geometry.bladeNumber

    return dT, dQ


def SolveBEM(solutionProperties: st.SolverData):
    """
	Iterates over annulus-center radial positions, solves the local induction
	factors, evaluates sectional loads, and integrates thrust and torque to
	obtain the total rotor performance.

	Args:
		solutionProperties: SolverData instance containing geometry, solver settings,
			and will be updated with results

	Returns:
		results are stored in solutionProperties.result
	"""
    geometry = solutionProperties.geometry

    annulusEdges = BuildAnnulusEdges(solutionProperties)
    radius = 0.5 * (annulusEdges[:-1] + annulusEdges[1:])
    annulusWidth = np.diff(annulusEdges)

    dT = np.array([])
    dQ = np.array([])

    thrust = 0.0
    torque = 0.0

    for index, radiusIterator in enumerate(radius):
        if DEBUG_PRINT:
            print("-" * 20)
        if DEBUG_PRINT:
            print(f"Solving for radius: {radiusIterator:.2f} m")

        dTCurrent, dQCurrent = solveElement(radiusIterator, annulusWidth[index], solutionProperties)

        dT = np.append(dT, dTCurrent)
        dQ = np.append(dQ, dQCurrent)

        thrust += dTCurrent * annulusWidth[index]
        torque += dQCurrent * annulusWidth[index]

        convergenceData = solutionProperties.getElementSolution(radiusIterator)
        if convergenceData:
            if DEBUG_PRINT:
                print(
                    f"Solution converged: {convergenceData.converged}, iterations: {convergenceData.iterations}, precision: {convergenceData.precision}")
        else:
            if DEBUG_PRINT:
                print(f"No convergence data found for radius: {radiusIterator:.2f} m")
        if DEBUG_PRINT:
            print(f"Differential thrust: {dTCurrent:.2f} N/m, Differential torque: {dQCurrent:.2f} Nm/m")

    # Calculate coefficient of power
    rotorArea = np.pi * geometry.tipRadius ** 2
    power = torque * geometry.rotationalSpeed  # Power = Torque * angular velocity
    windPower = 0.5 * solutionProperties.airDensity * rotorArea * geometry.freeStreamVelocity ** 3  # Available wind power
    cP = power / windPower if windPower > 0 else 0

    if solutionProperties.verboseOutput:
        print("=" * 40)
        print("Final results:")
        print(f"Total thrust: {thrust:.2f} N, Total torque: {torque:.2f} Nm")
        print(f"Total power: {power:.2f} W, Coefficient of power cP: {cP:.4f}")

    solutionProperties.result = st.Result(radius, dT, dQ, cP, thrust, torque, annulusWidth)

    return
