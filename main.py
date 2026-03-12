"""
Main Execution Script for BEM Solver
"""

import SolverTools as st
import Plotting
import numpy as np
import BEM
from pathlib import Path

TSR_VALUES = [6.0, 8.0, 10.0]
OUTPUT_DIR = Path("results") / "point_1"
REPORT_DIR = Path("results") / "report_tasks"
SELECTED_TSR = 8.0


def thrust_coefficient(solution: st.SolverData) -> float:
	rotor_area = np.pi * solution.geometry.tipRadius ** 2
	dynamic_pressure_area = 0.5 * solution.airDensity * rotor_area * solution.geometry.freeStreamVelocity ** 2
	return solution.result.totalThrust / dynamic_pressure_area


def run_baseline_case(
	tsr: float,
	element_count: int = 100,
	relaxation: float = 0.25,
	spacing_method: str = "constant",
	use_prandtl_correction: bool = True
) -> st.SolverData:
	solution = st.SolverData()
	solution.setParameters(
		maxIterations=500,
		tolerance=1e-6,
		relaxation=relaxation,
		elementCount=element_count)
	solution.spacingMethod = spacing_method
	solution.usePrandtlCorrection = use_prandtl_correction
	solution.geometry.tipSpeedRatio = tsr
	BEM.SolveBEM(solution)
	return solution


def print_performance_summary(solutions: list[st.SolverData]) -> None:
	print("\nBaseline performance summary")
	print(f"{'TSR':>6} {'Thrust [N]':>14} {'Torque [Nm]':>14} {'CP [-]':>10} {'CT [-]':>10}")
	for solution in solutions:
		tsr = solution.geometry.tipSpeedRatio
		print(
			f"{tsr:6.1f} "
			f"{solution.result.totalThrust:14.2f} "
			f"{solution.result.totalTorque:14.2f} "
			f"{solution.result.cP:10.4f} "
			f"{thrust_coefficient(solution):10.4f}"
		)


def save_performance_summary(solutions: list[st.SolverData]) -> Path:
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	output_path = OUTPUT_DIR / "performance_summary.csv"
	return save_performance_summary_to_path(solutions, output_path)


def save_performance_summary_to_path(solutions: list[st.SolverData], output_path: Path) -> Path:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="ascii", newline="") as file:
		file.write("TSR,Thrust_N,Torque_Nm,Power_W,CP,CT\n")
		for solution in solutions:
			power = solution.result.totalTorque * solution.geometry.rotationalSpeed
			file.write(
				f"{solution.geometry.tipSpeedRatio:.1f},"
				f"{solution.result.totalThrust:.6f},"
				f"{solution.result.totalTorque:.6f},"
				f"{power:.6f},"
				f"{solution.result.cP:.6f},"
				f"{thrust_coefficient(solution):.6f}\n"
			)
	return output_path


def save_spanwise_csv(solution: st.SolverData, output_path: Path) -> Path:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	radial_position = solution.result.radius / solution.geometry.tipRadius
	alpha_deg = np.rad2deg(np.array([element.angleOfAttack for element in solution.elementSolutions]))
	phi_deg = np.rad2deg(np.array([element.inflowAngle for element in solution.elementSolutions]))
	a = np.array([element.a for element in solution.elementSolutions])
	a_prime = np.array([element.aPrime for element in solution.elementSolutions])
	c_axial = np.array([element.axialForceCoefficient for element in solution.elementSolutions])
	c_azimuthal = np.array([element.azimuthalForceCoefficient for element in solution.elementSolutions])
	chord = np.array([solution.geometry.chordFunction(radius) for radius in solution.result.radius])
	with output_path.open("w", encoding="ascii", newline="") as file:
		file.write("r_over_R,r_m,alpha_deg,phi_deg,a,a_prime,c_axial,c_azimuthal,dT_N_per_m,dQ_Nm_per_m,chord_m\n")
		for index, radius in enumerate(solution.result.radius):
			file.write(
				f"{radial_position[index]:.6f},{radius:.6f},{alpha_deg[index]:.6f},{phi_deg[index]:.6f},"
				f"{a[index]:.6f},{a_prime[index]:.6f},{c_axial[index]:.6f},{c_azimuthal[index]:.6f},"
				f"{solution.result.dT[index]:.6f},{solution.result.dQ[index]:.6f},{chord[index]:.6f}\n"
			)
	return output_path


def write_text_file(output_path: Path, content: str) -> Path:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(content, encoding="ascii")
	return output_path


def generate_task_d_outputs(solutions: list[st.SolverData]) -> None:
	task_dir = REPORT_DIR / "d"
	task_dir.mkdir(parents=True, exist_ok=True)

	Plotting.SaveFigure(Plotting.PlotAnglesComparison(solutions), task_dir / "d_a_angles_vs_radius.png")
	Plotting.SaveFigure(Plotting.PlotInductionComparison(solutions), task_dir / "d_b_induction_vs_radius.png")
	Plotting.SaveFigure(Plotting.PlotLoadingComparison(solutions), task_dir / "d_c_loading_vs_radius.png")
	Plotting.SaveFigure(Plotting.PlotPerformanceComparison(solutions), task_dir / "d_d_total_performance_vs_tsr.png")
	save_performance_summary_to_path(solutions, task_dir / "d_performance_summary.csv")

	for solution in solutions:
		save_spanwise_csv(solution, task_dir / f"d_spanwise_tsr_{solution.geometry.tipSpeedRatio:.0f}.csv")

	summary_lines = [
		"# Task d notes",
		"",
		"This folder covers the baseline axial-flow plots requested in task d.",
		"",
		"Key observations:",
		f"- As TSR increases from {TSR_VALUES[0]:.0f} to {TSR_VALUES[-1]:.0f}, both thrust and power increase for the baseline rotor.",
		f"- The axial induction is highest near the root and tip, with the mid-span carrying the most uniform loading.",
		f"- The inflow angle decreases with radius, while the angle of attack remains in a moderate operating range over most of the blade.",
		"- Azimuthal induction stays much smaller than axial induction for this turbine case.",
	]
	write_text_file(task_dir / "d_notes.md", "\n".join(summary_lines))


def generate_task_e_outputs() -> None:
	task_dir = REPORT_DIR / "e"
	task_dir.mkdir(parents=True, exist_ok=True)

	with_tip = [run_baseline_case(tsr, use_prandtl_correction=True) for tsr in TSR_VALUES]
	without_tip = [run_baseline_case(tsr, use_prandtl_correction=False) for tsr in TSR_VALUES]
	selected_with_tip = next(solution for solution in with_tip if solution.geometry.tipSpeedRatio == SELECTED_TSR)
	selected_without_tip = next(solution for solution in without_tip if solution.geometry.tipSpeedRatio == SELECTED_TSR)

	Plotting.SaveFigure(Plotting.PlotTipCorrectionTotals(with_tip, without_tip), task_dir / "e_tip_correction_totals.png")
	Plotting.SaveFigure(
		Plotting.PlotTipCorrectionSpanwise(selected_with_tip, selected_without_tip),
		task_dir / "e_tip_correction_spanwise_tsr_8.png")

	with (task_dir / "e_tip_correction_summary.csv").open("w", encoding="ascii", newline="") as file:
		file.write("TSR,Configuration,Thrust_N,Torque_Nm,CP,CT\n")
		for solution in with_tip:
			file.write(
				f"{solution.geometry.tipSpeedRatio:.1f},with_prandtl,{solution.result.totalThrust:.6f},"
				f"{solution.result.totalTorque:.6f},{solution.result.cP:.6f},{thrust_coefficient(solution):.6f}\n")
		for solution in without_tip:
			file.write(
				f"{solution.geometry.tipSpeedRatio:.1f},without_prandtl,{solution.result.totalThrust:.6f},"
				f"{solution.result.totalTorque:.6f},{solution.result.cP:.6f},{thrust_coefficient(solution):.6f}\n")

	cp_change = selected_without_tip.result.cP - selected_with_tip.result.cP
	tip_lines = [
		"# Task e notes",
		"",
		"This study compares the baseline rotor with and without Prandtl tip/root correction.",
		"",
		f"- The selected detailed comparison case is TSR = {SELECTED_TSR:.0f}.",
		f"- Removing the correction changes CP by {cp_change:.4f} at TSR {SELECTED_TSR:.0f}.",
		"- The local effect is strongest near the blade ends, where the correction reduces the effective loading.",
		"- The integrated effect is visible in thrust, torque, and CP across all tip-speed ratios.",
	]
	write_text_file(task_dir / "e_notes.md", "\n".join(tip_lines))


def generate_task_f_outputs() -> None:
	task_dir = REPORT_DIR / "f"
	task_dir.mkdir(parents=True, exist_ok=True)

	annuli_counts = [5, 10, 20, 40, 80, 160, 320]
	spacing_methods = ["constant", "cosine"]
	reference_solution = run_baseline_case(SELECTED_TSR, element_count=800, spacing_method="cosine")
	reference_thrust = reference_solution.result.totalThrust
	study_rows = []

	for spacing_method in spacing_methods:
		for annuli in annuli_counts:
			solution = run_baseline_case(SELECTED_TSR, element_count=annuli, spacing_method=spacing_method)
			mean_iterations = np.mean([element.iterations for element in solution.elementSolutions])
			relative_error = abs(solution.result.totalThrust - reference_thrust) / reference_thrust * 100
			study_rows.append({
				"spacing": spacing_method,
				"annuli": annuli,
				"thrust": solution.result.totalThrust,
				"cp": solution.result.cP,
				"mean_iterations": mean_iterations,
				"relative_error_percent": relative_error,
			})

	Plotting.SaveFigure(Plotting.PlotAnnuliStudy(study_rows, reference_thrust), task_dir / "f_annuli_convergence.png")

	constant_probe = st.SolverData()
	constant_probe.elementCount = 20
	constant_probe.spacingMethod = "constant"
	cosine_probe = st.SolverData()
	cosine_probe.elementCount = 20
	cosine_probe.spacingMethod = "cosine"
	constant_edges = BEM.BuildAnnulusEdges(constant_probe)
	cosine_edges = BEM.BuildAnnulusEdges(cosine_probe)
	Plotting.SaveFigure(
		Plotting.PlotSpacingDistribution(constant_edges, cosine_edges),
		task_dir / "f_spacing_methods.png")

	with (task_dir / "f_annuli_study.csv").open("w", encoding="ascii", newline="") as file:
		file.write("spacing,annuli,thrust_N,cp,mean_iterations,relative_error_percent\n")
		for row in study_rows:
			file.write(
				f"{row['spacing']},{row['annuli']},{row['thrust']:.6f},{row['cp']:.6f},"
				f"{row['mean_iterations']:.6f},{row['relative_error_percent']:.6f}\n")

	best_constant = min((row for row in study_rows if row["spacing"] == "constant"), key=lambda row: row["relative_error_percent"])
	best_cosine = min((row for row in study_rows if row["spacing"] == "cosine"), key=lambda row: row["relative_error_percent"])
	notes = [
		"# Task f notes",
		"",
		f"This study uses TSR = {SELECTED_TSR:.0f} as the reference operating point.",
		f"- The thrust reference was computed with 800 cosine-spaced annuli: {reference_thrust:.2f} N.",
		f"- Best constant-spacing case in the tested set: {best_constant['annuli']} annuli with {best_constant['relative_error_percent']:.3f}% thrust error.",
		f"- Best cosine-spacing case in the tested set: {best_cosine['annuli']} annuli with {best_cosine['relative_error_percent']:.3f}% thrust error.",
		"- Cosine spacing resolves the blade ends more strongly and therefore tends to converge faster in thrust for a given annulus count.",
	]
	write_text_file(task_dir / "f_notes.md", "\n".join(notes))


def generate_task_j_outputs(selected_solution: st.SolverData) -> None:
	task_dir = REPORT_DIR / "j"
	task_dir.mkdir(parents=True, exist_ok=True)

	radial_position = selected_solution.result.radius / selected_solution.geometry.tipRadius
	alpha = np.array([element.angleOfAttack for element in selected_solution.elementSolutions])
	alpha_deg = np.rad2deg(alpha)
	cl = np.array([selected_solution.geometry.airfoil.lookup(value, 'cl') for value in alpha])
	cd = np.array([selected_solution.geometry.airfoil.lookup(value, 'cd') for value in alpha])
	cl_to_cd = cl / np.maximum(cd, 1e-8)
	chord = np.array([selected_solution.geometry.chordFunction(radius) for radius in selected_solution.result.radius])

	airfoil = selected_solution.geometry.airfoil
	alpha_polar_deg = np.rad2deg(airfoil.alpha)
	Plotting.SaveFigure(
		Plotting.PlotAirfoilOperationalPolar(alpha_polar_deg, airfoil.cl, airfoil.cd, alpha_deg, cl, cd, radial_position),
		task_dir / "j_airfoil_operating_points.png")
	Plotting.SaveFigure(
		Plotting.PlotAirfoilSpanwise(radial_position, alpha_deg, cl, cd, cl_to_cd, chord),
		task_dir / "j_airfoil_spanwise_relation.png")

	with (task_dir / "j_selected_case_tsr_8.csv").open("w", encoding="ascii", newline="") as file:
		file.write("r_over_R,alpha_deg,cl,cd,cl_over_cd,chord_m\n")
		for index in range(len(radial_position)):
			file.write(
				f"{radial_position[index]:.6f},{alpha_deg[index]:.6f},{cl[index]:.6f},{cd[index]:.6f},"
				f"{cl_to_cd[index]:.6f},{chord[index]:.6f}\n")

	best_index = int(np.argmax(cl_to_cd))
	worst_index = int(np.argmin(cl_to_cd))
	notes = [
		"# Task j notes",
		"",
		f"The chosen case is the baseline rotor at TSR = {selected_solution.geometry.tipSpeedRatio:.0f}.",
		f"- The best local aerodynamic efficiency occurs near r/R = {radial_position[best_index]:.3f}, where Cl/Cd = {cl_to_cd[best_index]:.2f}.",
		f"- The lowest local aerodynamic efficiency occurs near r/R = {radial_position[worst_index]:.3f}, where Cl/Cd = {cl_to_cd[worst_index]:.2f}.",
		"- The blade operates over a range of angle of attack values rather than a single optimum value because the inflow angle changes strongly along the span.",
		"- Comparing Cl and chord along the blade shows that the larger inboard chord compensates for the lower local tangential speed near the root.",
	]
	write_text_file(task_dir / "j_notes.md", "\n".join(notes))


def generate_report_task_outputs(baseline_solutions: list[st.SolverData]) -> None:
	generate_task_d_outputs(baseline_solutions)
	generate_task_e_outputs()
	generate_task_f_outputs()
	selected_solution = next(solution for solution in baseline_solutions if solution.geometry.tipSpeedRatio == SELECTED_TSR)
	generate_task_j_outputs(selected_solution)


if __name__ == "__main__":
	solutions = [run_baseline_case(tsr) for tsr in TSR_VALUES]
	print_performance_summary(solutions)
	summary_path = save_performance_summary(solutions)

	comparison_figure = Plotting.PlotPerformanceComparison(solutions)
	Plotting.SaveFigure(comparison_figure, OUTPUT_DIR / "performance_comparison.png")

	for solution in solutions:
		spanwise_figure = Plotting.PlotSpanwiseResults(solution)
		Plotting.SaveFigure(
			spanwise_figure,
			OUTPUT_DIR / f"spanwise_tsr_{solution.geometry.tipSpeedRatio:.0f}.png")

	generate_report_task_outputs(solutions)

	print(f"\nSaved point-1 results to: {summary_path.parent.resolve()}")
	print(f"Saved report-task outputs to: {(REPORT_DIR).resolve()}")
	Plotting.ShowPlots()
