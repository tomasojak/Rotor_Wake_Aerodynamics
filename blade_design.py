"""
Point 2 design workflow for the wind-turbine BEM assignment.

Stage A:
- redesign with collective pitch only
- enforce CT = 0.75 at TSR = 8

Stage B:
- redesign with twist scaling + collective pitch
- enforce CT = 0.75 at TSR = 8
- maximize CP over the tested twist-scale range
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import BEM
import Plotting
import SolverTools as st


TARGET_TSR = 8.0
TARGET_CT = 0.75
PITCH_BRACKET_EXPANSION_DEG = 2.0
MAX_BRACKET_EXPANSIONS = 8
STAGE_B_COARSE_SCALE_MIN = 0.75
STAGE_B_COARSE_SCALE_MAX = 1.25
STAGE_B_COARSE_SCALE_POINTS = 13
STAGE_B_FINE_WINDOW = 0.05
STAGE_C_COARSE_SCALE_MIN = 0.70
STAGE_C_COARSE_SCALE_MAX = 1.40
STAGE_C_COARSE_SCALE_POINTS = 15
STAGE_C_FINE_WINDOW = 0.05


def baseline_twist(geometry: st.Geometry, radius: float) -> float:
    return 14.0 * (1 - geometry.dimensionlessRadialPosition(radius))


def baseline_chord(geometry: st.Geometry, radius: float) -> float:
    return 3.0 * (1 - geometry.dimensionlessRadialPosition(radius)) + 1.0


def thrust_coefficient(solution: st.SolverData) -> float:
    rotor_area = np.pi * solution.geometry.tipRadius ** 2
    denominator = 0.5 * solution.airDensity * rotor_area * solution.geometry.freeStreamVelocity ** 2
    return solution.result.totalThrust / denominator


def actuator_disk_cp(ct_value: float) -> float:
    axial_induction = 0.5 * (1 - np.sqrt(1 - ct_value))
    return 4 * axial_induction * (1 - axial_induction) ** 2


@dataclass
class DesignEvaluation:
    label: str
    pitch_deg: float
    twist_scale: float
    chord_scale: float
    solution: st.SolverData
    # Cache scalar outputs for fast ranking/export without recomputing from nested objects.
    ct: float
    cp: float
    thrust: float
    torque: float


def spanwise_data(evaluation: DesignEvaluation) -> dict[str, np.ndarray]:
    solution = evaluation.solution
    radial = solution.result.radius / solution.geometry.tipRadius
    alpha = np.rad2deg(np.array([element.angleOfAttack for element in solution.elementSolutions]))
    phi = np.rad2deg(np.array([element.inflowAngle for element in solution.elementSolutions]))
    a = np.array([element.a for element in solution.elementSolutions])
    a_prime = np.array([element.aPrime for element in solution.elementSolutions])
    chord = np.array([solution.geometry.chordFunction(radius) for radius in solution.result.radius])
    twist = np.array([solution.geometry.bladeTwist(radius) for radius in solution.result.radius])
    return {
        "radial": radial,
        "alpha": alpha,
        "phi": phi,
        "a": a,
        "a_prime": a_prime,
        "dT": solution.result.dT,
        "dQ": solution.result.dQ,
        "chord": chord,
        "twist": twist,
    }


def evaluate_design(
    pitch_deg: float,
    twist_scale: float = 1.0,
    chord_scale: float = 1.0,
    label: str = "",
    element_count: int = 80,
    spacing_method: str = "cosine") -> DesignEvaluation:

    solution = st.SolverData()
    solution.setParameters(
        maxIterations=300,
        tolerance=1e-6,
        relaxation=0.25,
        elementCount=element_count)
    solution.verboseOutput = False
    solution.spacingMethod = spacing_method
    solution.geometry.tipSpeedRatio = TARGET_TSR
    solution.geometry.bladePitch = pitch_deg
    solution.geometry.bladeTwist = lambda radius, geometry=solution.geometry, scale=twist_scale: (
        scale * baseline_twist(geometry, radius)
    )
    solution.geometry.chordFunction = lambda radius, geometry=solution.geometry, scale=chord_scale: (
        scale * baseline_chord(geometry, radius)
    )

    BEM.SolveBEM(solution)

    return DesignEvaluation(
        label=label,
        pitch_deg=pitch_deg,
        twist_scale=twist_scale,
        chord_scale=chord_scale,
        solution=solution,
        ct=thrust_coefficient(solution),
        cp=solution.result.cP,
        thrust=solution.result.totalThrust,
        torque=solution.result.totalTorque)


def solve_pitch_for_target_ct(
    target_ct: float,
    twist_scale: float = 1.0,
    chord_scale: float = 1.0,
    pitch_bounds: tuple[float, float] = (-12.0, 8.0),
    tolerance: float = 1e-4,
    max_iterations: int = 24,
    label_prefix: str = "",
    element_count: int = 80
) -> DesignEvaluation:
    lower_pitch, upper_pitch = pitch_bounds
    lower_eval = evaluate_design(
        lower_pitch, twist_scale, chord_scale, label=f"{label_prefix}_lower", element_count=element_count)
    upper_eval = evaluate_design(
        upper_pitch, twist_scale, chord_scale, label=f"{label_prefix}_upper", element_count=element_count)

    # Expand the initial pitch bounds until CT(target) is bracketed for robust bisection.
    expand_counter = 0
    while not (lower_eval.ct >= target_ct >= upper_eval.ct):
        if expand_counter > MAX_BRACKET_EXPANSIONS:
            raise RuntimeError(
                f"Could not bracket CT={target_ct:.3f} for twist scale {twist_scale:.3f}. "
                f"Bounds gave CT {lower_eval.ct:.4f} and {upper_eval.ct:.4f}."
            )
        lower_pitch -= PITCH_BRACKET_EXPANSION_DEG
        upper_pitch += PITCH_BRACKET_EXPANSION_DEG
        lower_eval = evaluate_design(
            lower_pitch, twist_scale, chord_scale, label=f"{label_prefix}_lower", element_count=element_count)
        upper_eval = evaluate_design(
            upper_pitch, twist_scale, chord_scale, label=f"{label_prefix}_upper", element_count=element_count)
        expand_counter += 1

    best_eval = lower_eval if abs(lower_eval.ct - target_ct) < abs(upper_eval.ct - target_ct) else upper_eval

    for _ in range(max_iterations):
        mid_pitch = 0.5 * (lower_pitch + upper_pitch)
        mid_eval = evaluate_design(mid_pitch, twist_scale, chord_scale, label=f"{label_prefix}_mid", element_count=element_count)
        if abs(mid_eval.ct - target_ct) < abs(best_eval.ct - target_ct):
            best_eval = mid_eval

        if abs(mid_eval.ct - target_ct) < tolerance:
            return mid_eval

        if mid_eval.ct > target_ct:
            lower_pitch = mid_pitch
            lower_eval = mid_eval
        else:
            upper_pitch = mid_pitch
            upper_eval = mid_eval

    return best_eval


def run_stage_a() -> DesignEvaluation:
    stage_a_search = solve_pitch_for_target_ct(
        target_ct=TARGET_CT,
        twist_scale=1.0,
        chord_scale=1.0,
        label_prefix="stage_a_pitch_only",
        element_count=80)
    return evaluate_design(
        stage_a_search.pitch_deg, twist_scale=1.0, chord_scale=1.0, label="stage_a_pitch_only", element_count=120)


def run_stage_b() -> tuple[DesignEvaluation, list[DesignEvaluation]]:
    # Coarse scan around baseline twist scale=1.0 to identify a promising region before local refinement.
    coarse_scales = np.linspace(STAGE_B_COARSE_SCALE_MIN, STAGE_B_COARSE_SCALE_MAX, STAGE_B_COARSE_SCALE_POINTS)
    coarse_results = [
        solve_pitch_for_target_ct(
            TARGET_CT,
            twist_scale=scale,
            chord_scale=1.0,
            label_prefix=f"stage_b_coarse_{index}",
            element_count=60)
        for index, scale in enumerate(coarse_scales)
    ]
    coarse_best = max(coarse_results, key=lambda evaluation: evaluation.cp)

    fine_min = max(0.60, coarse_best.twist_scale - STAGE_B_FINE_WINDOW)
    fine_max = min(1.40, coarse_best.twist_scale + STAGE_B_FINE_WINDOW)
    fine_scales = np.linspace(fine_min, fine_max, 17)
    fine_results = [
        solve_pitch_for_target_ct(
            TARGET_CT,
            twist_scale=scale,
            chord_scale=1.0,
            label_prefix=f"stage_b_fine_{index}")
        for index, scale in enumerate(fine_scales)
    ]

    all_results = coarse_results + fine_results
    best_search = max(all_results, key=lambda evaluation: evaluation.cp)
    final_best = evaluate_design(
        best_search.pitch_deg,
        twist_scale=best_search.twist_scale,
        chord_scale=1.0,
        label="stage_b_twist_plus_pitch",
        element_count=120)
    return final_best, all_results


def run_stage_c() -> tuple[DesignEvaluation, list[DesignEvaluation]]:
    # Coarse chord-scaling scan followed by local refinement around the best candidate.
    coarse_scales = np.linspace(STAGE_C_COARSE_SCALE_MIN, STAGE_C_COARSE_SCALE_MAX, STAGE_C_COARSE_SCALE_POINTS)
    coarse_results: list[DesignEvaluation] = []
    for index, scale in enumerate(coarse_scales):
        try:
            coarse_results.append(
                solve_pitch_for_target_ct(
                    TARGET_CT,
                    twist_scale=1.0,
                    chord_scale=scale,
                    label_prefix=f"stage_c_coarse_{index}",
                    element_count=60))
        except RuntimeError:
            continue

    if not coarse_results:
        raise RuntimeError("Stage C failed: no feasible chord scale could satisfy CT = 0.75.")

    coarse_best = max(coarse_results, key=lambda evaluation: evaluation.cp)

    fine_min = max(0.50, coarse_best.chord_scale - STAGE_C_FINE_WINDOW)
    fine_max = min(2.00, coarse_best.chord_scale + STAGE_C_FINE_WINDOW)
    fine_scales = np.linspace(fine_min, fine_max, 17)
    fine_results: list[DesignEvaluation] = []
    for index, scale in enumerate(fine_scales):
        try:
            fine_results.append(
                solve_pitch_for_target_ct(
                    TARGET_CT,
                    twist_scale=1.0,
                    chord_scale=scale,
                    label_prefix=f"stage_c_fine_{index}"))
        except RuntimeError:
            continue

    all_results = coarse_results + fine_results
    if not all_results:
        raise RuntimeError("Stage C failed: no feasible coarse/fine chord candidate found.")

    best_search = max(all_results, key=lambda evaluation: evaluation.cp)
    # Re-solve pitch at final resolution so exported Case C hits CT target tightly.
    final_best = solve_pitch_for_target_ct(
        TARGET_CT,
        twist_scale=1.0,
        chord_scale=best_search.chord_scale,
        tolerance=1e-6,
        max_iterations=40,
        label_prefix="stage_c_final",
        element_count=120)
    final_best.label = "stage_c_chord_plus_pitch"
    return final_best, all_results


def compute_point_2_designs() -> tuple[
    DesignEvaluation,
    DesignEvaluation,
    DesignEvaluation,
    DesignEvaluation,
    list[DesignEvaluation],
    list[DesignEvaluation]
]:
    baseline = evaluate_design(pitch_deg=-2.0, twist_scale=1.0, chord_scale=1.0, label="baseline_tsr8", element_count=120)
    stage_a = run_stage_a()
    stage_a.label = "stage_a_pitch_only"
    stage_b, stage_b_results = run_stage_b()
    stage_b.label = "stage_b_twist_plus_pitch"
    stage_c, stage_c_results = run_stage_c()
    stage_c.label = "stage_c_chord_plus_pitch"
    return baseline, stage_a, stage_b, stage_c, stage_b_results, stage_c_results


def select_final_redesign(stage_b: DesignEvaluation, stage_c: DesignEvaluation) -> DesignEvaluation:
    return stage_b if stage_b.cp >= stage_c.cp else stage_c


def stagnation_pressure_profiles(evaluation: DesignEvaluation) -> dict[str, np.ndarray]:
    solution = evaluation.solution
    data = spanwise_data(evaluation)
    free_stream_velocity = solution.geometry.freeStreamVelocity
    air_density = solution.airDensity
    dynamic_pressure = 0.5 * air_density * free_stream_velocity ** 2
    a = data["a"]

    wake_velocity = free_stream_velocity * np.maximum(1 - 2 * a, 0.0)

    p0_upstream = np.full_like(a, dynamic_pressure)
    p0_before_disk = np.full_like(a, dynamic_pressure)
    p0_after_disk = 0.5 * air_density * wake_velocity ** 2
    p0_far_wake = p0_after_disk.copy()

    return {
        "radial": data["radial"],
        "p0_upstream": p0_upstream,
        "p0_before_disk": p0_before_disk,
        "p0_after_disk": p0_after_disk,
        "p0_far_wake": p0_far_wake,
        "p0_upstream_star": p0_upstream / dynamic_pressure,
        "p0_before_disk_star": p0_before_disk / dynamic_pressure,
        "p0_after_disk_star": p0_after_disk / dynamic_pressure,
        "p0_far_wake_star": p0_far_wake / dynamic_pressure,
    }

def generate_task_ghi_outputs(point2_bundle, tasks_dir: Path) -> list[str]:
    baseline, stage_a, stage_b, stage_c, stage_b_results, stage_c_results = point2_bundle
    final_redesign = select_final_redesign(stage_b, stage_c)
    tasks_dir.mkdir(parents=True, exist_ok=True)

    Plotting.SaveFigure(Plotting.PlotDesignSearchVsTwist(stage_b_results), tasks_dir / "g_design_search.png")
    Plotting.SaveFigure(Plotting.PlotDesignSearchVsChord(stage_c_results), tasks_dir / "g_case_c_design_search.png")
    Plotting.SaveFigure(Plotting.PlotFinalDesignGeometry(baseline, final_redesign), tasks_dir / "h_geometry_comparison.png")
    Plotting.SaveFigure(Plotting.PlotFinalDesignPerformance(baseline, final_redesign), tasks_dir / "h_performance_comparison.png")
    Plotting.SaveFigure(Plotting.PlotFinalDesignOperatingPoints(baseline, final_redesign), tasks_dir / "h_operating_points.png")

    baseline_pressure = stagnation_pressure_profiles(baseline)
    redesign_pressure = stagnation_pressure_profiles(final_redesign)
    Plotting.SaveFigure(
        Plotting.PlotStagnationPressureDistribution(baseline_pressure, redesign_pressure),
        tasks_dir / "i_stagnation_pressure_distribution.png")

    return [
        "## Task g",
        f"- Fixed TSR = {TARGET_TSR:.0f}, constrained CT target = {TARGET_CT:.2f}.",
        "- Stage A: pitch-only constraint solve.",
        "- Stage B: twist-scale scan + pitch trim for CT target.",
        "- Stage C: chord-scale scan + pitch trim for CT target.",
        f"- Final selected redesign by CP: {final_redesign.label}.",
        "",
        "## Task h",
        f"- Baseline CP at TSR 8: {baseline.cp:.4f}.",
        f"- Stage A CP at CT=0.75: {stage_a.cp:.4f}.",
        f"- Stage B CP at CT=0.75: {stage_b.cp:.4f}.",
        f"- Stage C CP at CT=0.75: {stage_c.cp:.4f}.",
        f"- Final redesign CP improvement over baseline: {final_redesign.cp - baseline.cp:.4f}.",
        "",
        "## Task i",
        "- Plotted normalized stagnation pressure at four actuator-disk stations.",
        "- Stations 1 and 2 overlap in ideal theory (no upstream energy extraction).",
        "- Stations 3 and 4 overlap in ideal theory (stagnation-pressure loss occurs at disk and remains downstream).",
    ]


def run_point_2() -> tuple[
    DesignEvaluation,
    DesignEvaluation,
    DesignEvaluation,
    DesignEvaluation,
    list[DesignEvaluation],
    list[DesignEvaluation]
]:
    baseline, stage_a, stage_b, stage_c, stage_b_results, stage_c_results = compute_point_2_designs()

    print("\nPoint 2 design summary")
    print(f"{'Case':<24} {'Pitch [deg]':>12} {'Twist Scale':>12} {'Chord Scale':>12} {'CT [-]':>10} {'CP [-]':>10}")
    for evaluation in [baseline, stage_a, stage_b, stage_c]:
        print(
            f"{evaluation.label:<24} "
            f"{evaluation.pitch_deg:12.4f} "
            f"{evaluation.twist_scale:12.4f} "
            f"{evaluation.chord_scale:12.4f} "
            f"{evaluation.ct:10.4f} "
            f"{evaluation.cp:10.4f}"
        )
    print(f"{'actuator_disk':<24} {'-':>12} {'-':>12} {'-':>12} {TARGET_CT:10.4f} {actuator_disk_cp(TARGET_CT):10.4f}")
    return baseline, stage_a, stage_b, stage_c, stage_b_results, stage_c_results


if __name__ == "__main__":
    run_point_2()
