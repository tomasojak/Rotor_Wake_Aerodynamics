"""
Main execution script for assignment outputs.
Running this file generates:
- task figures + one notes file
"""

import shutil
from pathlib import Path

import numpy as np

import BEM
import Plotting
import SolverTools as st
import blade_design

TSR_VALUES = [6.0, 8.0, 10.0]
SELECTED_TSR = 8.0
RESULTS_DIR = Path("results")
TASKS_DIR = Path("results") / "tasks"


def thrust_coefficient(solution: st.SolverData) -> float:
	rotor_area = np.pi * solution.geometry.tipRadius ** 2
	dynamic_pressure_area = 0.5 * solution.airDensity * rotor_area * solution.geometry.freeStreamVelocity ** 2
	return solution.result.totalThrust / dynamic_pressure_area


def run_baseline_case(
	tsr: float,
	element_count: int = 100,
	relaxation: float = 0.25,
	spacing_method: str = "constant",
	use_prandtl_correction: bool = True,
	verbose_output: bool = False
) -> st.SolverData:
	solution = st.SolverData()
	solution.setParameters(
		maxIterations=500,
		tolerance=1e-6,
		relaxation=relaxation,
		elementCount=element_count)
	solution.spacingMethod = spacing_method
	solution.usePrandtlCorrection = use_prandtl_correction
	solution.verboseOutput = verbose_output
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


def write_text_file(output_path: Path, content: str) -> Path:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(content, encoding="ascii")
	return output_path


def reset_output_dir(output_dir: Path) -> None:
	if output_dir.exists():
		for item in output_dir.iterdir():
			if item.is_file():
				item.unlink()
			elif item.is_dir():
				shutil.rmtree(item)
	output_dir.mkdir(parents=True, exist_ok=True)


def reset_results_root() -> None:
	if RESULTS_DIR.exists():
		for item in RESULTS_DIR.iterdir():
			if item.is_file():
				item.unlink()
			elif item.is_dir():
				shutil.rmtree(item)
	RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_task_d_outputs(solutions: list[st.SolverData]) -> list[str]:
	Plotting.SaveFigure(Plotting.PlotAnglesComparison(solutions), TASKS_DIR / "d_a_angles_vs_radius.png")
	Plotting.SaveFigure(Plotting.PlotInductionComparison(solutions), TASKS_DIR / "d_b_induction_vs_radius.png")
	Plotting.SaveFigure(Plotting.PlotLoadingComparison(solutions), TASKS_DIR / "d_c_loading_vs_radius.png")
	Plotting.SaveFigure(Plotting.PlotPerformanceComparison(solutions), TASKS_DIR / "d_d_total_performance_vs_tsr.png")

	return [
		"## Task d",
		f"- As TSR increases from {TSR_VALUES[0]:.0f} to {TSR_VALUES[-1]:.0f}, thrust and power increase for the baseline rotor.",
		"- Axial induction is strongest near root and tip; the mid-span is more uniform.",
		"- Inflow angle decreases with radius while angle of attack remains moderate over most of the blade.",
		"- Azimuthal induction remains smaller than axial induction in this baseline case.",
	]


def generate_task_e_outputs() -> list[str]:
	with_tip = [run_baseline_case(tsr, use_prandtl_correction=True) for tsr in TSR_VALUES]
	without_tip = [run_baseline_case(tsr, use_prandtl_correction=False) for tsr in TSR_VALUES]
	selected_with_tip = next(solution for solution in with_tip if solution.geometry.tipSpeedRatio == SELECTED_TSR)
	selected_without_tip = next(solution for solution in without_tip if solution.geometry.tipSpeedRatio == SELECTED_TSR)

	Plotting.SaveFigure(Plotting.PlotTipCorrectionTotals(with_tip, without_tip), TASKS_DIR / "e_tip_correction_totals.png")
	Plotting.SaveFigure(
		Plotting.PlotTipCorrectionSpanwise(selected_with_tip, selected_without_tip),
		TASKS_DIR / "e_tip_correction_spanwise_tsr_8.png")

	cp_change = selected_without_tip.result.cP - selected_with_tip.result.cP
	return [
		"## Task e",
		"This study compares the baseline rotor with and without Prandtl tip/root correction.",
		f"- Selected detailed case: TSR = {SELECTED_TSR:.0f}.",
		f"- Removing the correction changes CP by {cp_change:.4f} at TSR {SELECTED_TSR:.0f}.",
		"- The local effect is strongest near blade ends where finite-blade losses are concentrated.",
	]


def generate_task_f_outputs() -> list[str]:
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

	Plotting.SaveFigure(Plotting.PlotAnnuliStudy(study_rows, reference_thrust), TASKS_DIR / "f_annuli_convergence.png")

	constant_probe = st.SolverData()
	constant_probe.elementCount = 20
	constant_probe.spacingMethod = "constant"
	cosine_probe = st.SolverData()
	cosine_probe.elementCount = 20
	cosine_probe.spacingMethod = "cosine"
	constant_edges = BEM.BuildAnnulusEdges(constant_probe)
	cosine_edges = BEM.BuildAnnulusEdges(cosine_probe)
	Plotting.SaveFigure(Plotting.PlotSpacingDistribution(constant_edges, cosine_edges), TASKS_DIR / "f_spacing_methods.png")

	best_constant = min((row for row in study_rows if row["spacing"] == "constant"), key=lambda row: row["relative_error_percent"])
	best_cosine = min((row for row in study_rows if row["spacing"] == "cosine"), key=lambda row: row["relative_error_percent"])
	return [
		"## Task f",
		f"- Reference thrust uses 800 cosine annuli at TSR {SELECTED_TSR:.0f}: {reference_thrust:.2f} N.",
		f"- Best constant-spacing case: {best_constant['annuli']} annuli, error = {best_constant['relative_error_percent']:.3f}%.",
		f"- Best cosine-spacing case: {best_cosine['annuli']} annuli, error = {best_cosine['relative_error_percent']:.3f}%.",
		"- Cosine spacing converges faster for the same annulus count due to stronger end-region resolution.",
	]


def generate_task_j_outputs(selected_solution: st.SolverData) -> list[str]:
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
		TASKS_DIR / "j_airfoil_operating_points.png")
	Plotting.SaveFigure(
		Plotting.PlotAirfoilSpanwise(radial_position, alpha_deg, cl, cd, cl_to_cd, chord),
		TASKS_DIR / "j_airfoil_spanwise_relation.png")

	best_index = int(np.argmax(cl_to_cd))
	worst_index = int(np.argmin(cl_to_cd))
	return [
		"## Task j",
		f"- Selected case: baseline rotor at TSR = {selected_solution.geometry.tipSpeedRatio:.0f}.",
		f"- Best local Cl/Cd at r/R = {radial_position[best_index]:.3f}: {cl_to_cd[best_index]:.2f}.",
		f"- Lowest local Cl/Cd at r/R = {radial_position[worst_index]:.3f}: {cl_to_cd[worst_index]:.2f}.",
		"- Larger inboard chord compensates for lower local tangential speed near root.",
	]


def write_tasks_notes(sections: list[list[str]]) -> None:
	notes_lines = ["# Tasks notes", ""]
	for section in sections:
		notes_lines.extend(section)
		notes_lines.append("")

	write_text_file(TASKS_DIR / "notes.md", "\n".join(notes_lines).strip() + "\n")


if __name__ == "__main__":
	reset_results_root()
	reset_output_dir(TASKS_DIR)
	solutions = [run_baseline_case(tsr) for tsr in TSR_VALUES]
	print_performance_summary(solutions)
	selected_solution = next(solution for solution in solutions if solution.geometry.tipSpeedRatio == SELECTED_TSR)

	task_sections: list[list[str]] = []
	task_sections.append(generate_task_d_outputs(solutions))
	task_sections.append(generate_task_e_outputs())
	task_sections.append(generate_task_f_outputs())

	point_2_bundle = blade_design.run_point_2()
	task_sections.append(blade_design.generate_task_ghi_outputs(point_2_bundle, TASKS_DIR))
	task_sections.append(generate_task_j_outputs(selected_solution))
	write_tasks_notes(task_sections)

	Plotting.ShowPlots()
