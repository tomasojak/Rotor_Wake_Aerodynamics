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

import matplotlib.pyplot as plt
import numpy as np

import BEM
import Plotting
import SolverTools as st


TARGET_TSR = 8.0
TARGET_CT = 0.75
OUTPUT_DIR = Path("results") / "point_2"
REPORT_DIR = Path("results") / "report_tasks"


def baseline_twist(geometry: st.Geometry, radius: float) -> float:
    return 14.0 * (1 - geometry.dimensionlessRadialPosition(radius))


def baseline_chord(geometry: st.Geometry, radius: float) -> float:
    return 3.0 * (1 - geometry.dimensionlessRadialPosition(radius)) + 1.0


def thrust_coefficient(solution: st.SolverData) -> float: # FIX: feel free to make this a method of SolverData
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
    solution: st.SolverData
    ct: float
    cp: float
    thrust: float
    torque: float


def spanwise_data(evaluation: DesignEvaluation) -> dict[str, np.ndarray]: # FIX: could be a method of design evaluation
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
    twist_scale: float = 1.0, # FIX: No need to clutter with type hints that are obvious
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
    solution.geometry.chordFunction = lambda radius, geometry=solution.geometry: baseline_chord(geometry, radius)

    BEM.SolveBEM(solution)

    return DesignEvaluation(
        label=label,
        pitch_deg=pitch_deg,
        twist_scale=twist_scale,
        solution=solution,
        ct=thrust_coefficient(solution),
        cp=solution.result.cP, # FIX: why are you storing solution and then also its member variables separately?
        thrust=solution.result.totalThrust,
        torque=solution.result.totalTorque)


def solve_pitch_for_target_ct(
    target_ct: float,
    twist_scale: float = 1.0,
    pitch_bounds: tuple[float, float] = (-12.0, 8.0),
    tolerance: float = 1e-4,
    max_iterations: int = 24,
    label_prefix: str = "",
    element_count: int = 80
) -> DesignEvaluation:
    lower_pitch, upper_pitch = pitch_bounds
    lower_eval = evaluate_design(lower_pitch, twist_scale, label=f"{label_prefix}_lower", element_count=element_count)
    upper_eval = evaluate_design(upper_pitch, twist_scale, label=f"{label_prefix}_upper", element_count=element_count)

    expand_counter = 0 # FIX: Thi smight need a comment, I do not understand what it does
    while not (lower_eval.ct >= target_ct >= upper_eval.ct):
        if expand_counter > 8:
            raise RuntimeError(
                f"Could not bracket CT={target_ct:.3f} for twist scale {twist_scale:.3f}. "
                f"Bounds gave CT {lower_eval.ct:.4f} and {upper_eval.ct:.4f}."
            )
        # FIX: Am I reading it right that the 'bounds' are actually the initial values? You seem to increase the bounds on each pass
        lower_pitch -= 2.0 # FIX: Explain why those numbers were chosen
        upper_pitch += 2.0
        lower_eval = evaluate_design(lower_pitch, twist_scale, label=f"{label_prefix}_lower", element_count=element_count)
        upper_eval = evaluate_design(upper_pitch, twist_scale, label=f"{label_prefix}_upper", element_count=element_count)
        expand_counter += 1

    best_eval = lower_eval if abs(lower_eval.ct - target_ct) < abs(upper_eval.ct - target_ct) else upper_eval

    for _ in range(max_iterations):
        mid_pitch = 0.5 * (lower_pitch + upper_pitch)
        mid_eval = evaluate_design(mid_pitch, twist_scale, label=f"{label_prefix}_mid", element_count=element_count)
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
        label_prefix="stage_a_pitch_only",
        element_count=80)
    return evaluate_design(stage_a_search.pitch_deg, twist_scale=1.0, label="stage_a_pitch_only", element_count=120)


def run_stage_b() -> tuple[DesignEvaluation, list[DesignEvaluation]]:
    coarse_scales = np.linspace(0.75, 1.25, 13) # FIX: Explain how those numbers were chosen, and why 13 points
    coarse_results = [
        solve_pitch_for_target_ct(
            TARGET_CT,
            twist_scale=scale,
            label_prefix=f"stage_b_coarse_{index}",
            element_count=60)
        for index, scale in enumerate(coarse_scales)
    ]
    coarse_best = max(coarse_results, key=lambda evaluation: evaluation.cp)

    fine_min = max(0.60, coarse_best.twist_scale - 0.05)
    fine_max = min(1.40, coarse_best.twist_scale + 0.05)
    fine_scales = np.linspace(fine_min, fine_max, 17)
    fine_results = [
        solve_pitch_for_target_ct(
            TARGET_CT,
            twist_scale=scale,
            label_prefix=f"stage_b_fine_{index}",
            element_count=80) # FIX: 80 is the default, no need to add noise
        for index, scale in enumerate(fine_scales)
    ]

    all_results = coarse_results + fine_results
    best_search = max(all_results, key=lambda evaluation: evaluation.cp)
    final_best = evaluate_design(
        best_search.pitch_deg,
        twist_scale=best_search.twist_scale,
        label="stage_b_twist_plus_pitch",
        element_count=120)
    return final_best, all_results


def save_design_summary(path: Path, evaluations: list[DesignEvaluation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="") as file:
        file.write("label,pitch_deg,twist_scale,CT,CP,Thrust_N,Torque_Nm\n")
        for evaluation in evaluations:
            file.write(
                f"{evaluation.label},{evaluation.pitch_deg:.6f},{evaluation.twist_scale:.6f},"
                f"{evaluation.ct:.6f},{evaluation.cp:.6f},{evaluation.thrust:.6f},{evaluation.torque:.6f}\n"
            )


def save_stage_b_scan(path: Path, evaluations: list[DesignEvaluation]) -> None:
    ordered = sorted(evaluations, key=lambda evaluation: evaluation.twist_scale)
    save_design_summary(path, ordered)


def save_spanwise_design_csv(path: Path, evaluation: DesignEvaluation) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    solution = evaluation.solution
    data = spanwise_data(evaluation)

    with path.open("w", encoding="ascii", newline="") as file:
        file.write("r_over_R,alpha_deg,phi_deg,a,a_prime,dT_N_per_m,dQ_Nm_per_m,chord_m,twist_deg\n")
        for index in range(len(data["radial"])):
            file.write(
                f"{data['radial'][index]:.6f},{data['alpha'][index]:.6f},{data['phi'][index]:.6f},"
                f"{data['a'][index]:.6f},{data['a_prime'][index]:.6f},{solution.result.dT[index]:.6f},"
                f"{solution.result.dQ[index]:.6f},{data['chord'][index]:.6f},{data['twist'][index]:.6f}\n"
            )

# FIX: There is a separate file for plotting, use it.
# What follows is 400 lines that could be spared from this file
# It is much easier to look at the logic and then go to the plotting file to see how it is plotted
def plot_cp_vs_twist_scale(stage_b_results: list[DesignEvaluation]):
    ordered = sorted(stage_b_results, key=lambda evaluation: evaluation.twist_scale)
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


def plot_design_comparison(baseline: DesignEvaluation, stage_a: DesignEvaluation, stage_b: DesignEvaluation):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Point 2: Baseline vs Stage A vs Stage B at TSR = 8')

    evaluations = [baseline, stage_a, stage_b]
    labels = ['Baseline', 'Stage A', 'Stage B']

    for evaluation, label in zip(evaluations, labels):
        solution = evaluation.solution
        radial = solution.result.radius / solution.geometry.tipRadius
        alpha = np.rad2deg(np.array([element.angleOfAttack for element in solution.elementSolutions]))
        axs[0, 0].plot(radial, alpha, linewidth=2, label=label)
        axs[0, 1].plot(radial, np.array([element.a for element in solution.elementSolutions]), linewidth=2, label=label)
        axs[1, 0].plot(radial, solution.result.dT, linewidth=2, label=label)
        axs[1, 1].plot(radial, solution.result.dQ, linewidth=2, label=label)

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


def plot_design_distributions(baseline: DesignEvaluation, stage_a: DesignEvaluation, stage_b: DesignEvaluation):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle('Point 2: Blade-Design Distributions')

    evaluations = [baseline, stage_a, stage_b]
    labels = ['Baseline', 'Stage A', 'Stage B']
    for evaluation, label in zip(evaluations, labels):
        radial = evaluation.solution.result.radius / evaluation.solution.geometry.tipRadius
        twist = np.array([evaluation.solution.geometry.bladeTwist(radius) for radius in evaluation.solution.result.radius])
        chord = np.array([evaluation.solution.geometry.chordFunction(radius) for radius in evaluation.solution.result.radius])
        axs[0].plot(radial, twist, linewidth=2, label=label)
        axs[1].plot(radial, chord, linewidth=2, label=label)

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


def plot_design_summary_table(baseline: DesignEvaluation, stage_a: DesignEvaluation, stage_b: DesignEvaluation):
    actuator_cp = actuator_disk_cp(TARGET_CT)
    rows = [
        ['Baseline', f'{baseline.pitch_deg:.3f}', f'{baseline.twist_scale:.3f}', f'{baseline.ct:.4f}', f'{baseline.cp:.4f}'],
        ['Stage A', f'{stage_a.pitch_deg:.3f}', f'{stage_a.twist_scale:.3f}', f'{stage_a.ct:.4f}', f'{stage_a.cp:.4f}'],
        ['Stage B', f'{stage_b.pitch_deg:.3f}', f'{stage_b.twist_scale:.3f}', f'{stage_b.ct:.4f}', f'{stage_b.cp:.4f}'],
        ['Actuator disk', '-', '-', f'{TARGET_CT:.4f}', f'{actuator_cp:.4f}'],
    ]

    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.axis('off')
    table = ax.table(
        cellText=rows,
        colLabels=['Case', 'Pitch [deg]', 'Twist Scale [-]', 'CT [-]', 'CP [-]'],
        loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    fig.tight_layout()
    return fig


def write_notes(path: Path, baseline: DesignEvaluation, stage_a: DesignEvaluation, stage_b: DesignEvaluation) -> None:
    actuator_cp = actuator_disk_cp(TARGET_CT)
    lines = [
        "# Point 2 notes",
        "",
        f"Target operating point: TSR = {TARGET_TSR:.0f}, CT = {TARGET_CT:.2f}.",
        "",
        "Stage A: collective pitch only",
        f"- Required pitch: {stage_a.pitch_deg:.3f} deg",
        f"- Achieved CT: {stage_a.ct:.4f}",
        f"- Achieved CP: {stage_a.cp:.4f}",
        "",
        "Stage B: twist scaling plus collective pitch",
        f"- Best twist scale: {stage_b.twist_scale:.4f}",
        f"- Required pitch: {stage_b.pitch_deg:.3f} deg",
        f"- Achieved CT: {stage_b.ct:.4f}",
        f"- Achieved CP: {stage_b.cp:.4f}",
        "",
        "Comparison",
        f"- Baseline CP at TSR 8: {baseline.cp:.4f}",
        f"- Stage A CP improvement over baseline: {stage_a.cp - baseline.cp:.4f}",
        f"- Stage B CP improvement over baseline: {stage_b.cp - baseline.cp:.4f}",
        f"- Ideal actuator-disk CP at CT = 0.75: {actuator_cp:.4f}",
        f"- Stage B gap to actuator-disk CP: {actuator_cp - stage_b.cp:.4f}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="ascii")


def compute_point_2_designs() -> tuple[DesignEvaluation, DesignEvaluation, DesignEvaluation, list[DesignEvaluation]]:
    baseline = evaluate_design(pitch_deg=-2.0, twist_scale=1.0, label="baseline_tsr8", element_count=120)
    stage_a = run_stage_a()
    stage_a.label = "stage_a_pitch_only"
    stage_b, stage_b_results = run_stage_b()
    stage_b.label = "stage_b_twist_plus_pitch"
    return baseline, stage_a, stage_b, stage_b_results


def plot_final_design_performance(baseline: DesignEvaluation, redesign: DesignEvaluation):
    baseline_data = spanwise_data(baseline)
    redesign_data = spanwise_data(redesign)

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
        axis.plot(baseline_data["radial"], baseline_data[key], linewidth=2, label='Baseline')
        axis.plot(redesign_data["radial"], redesign_data[key], linewidth=2, label='Final redesign')
        axis.set_xlabel('Radial Position (r/R)')
        axis.set_ylabel(ylabel)
        axis.grid(True)
        axis.legend()

    fig.tight_layout()
    return fig


def plot_final_design_geometry(baseline: DesignEvaluation, redesign: DesignEvaluation):
    baseline_data = spanwise_data(baseline)
    redesign_data = spanwise_data(redesign)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle('Task h: Baseline vs Final Redesign Geometry')

    axs[0].plot(baseline_data["radial"], baseline_data["twist"], linewidth=2, label='Baseline')
    axs[0].plot(redesign_data["radial"], redesign_data["twist"], linewidth=2, label='Final redesign')
    axs[0].set_xlabel('Radial Position (r/R)')
    axs[0].set_ylabel('Twist [deg]')
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(baseline_data["radial"], baseline_data["chord"], linewidth=2, label='Baseline')
    axs[1].plot(redesign_data["radial"], redesign_data["chord"], linewidth=2, label='Final redesign')
    axs[1].set_xlabel('Radial Position (r/R)')
    axs[1].set_ylabel('Chord [m]')
    axs[1].grid(True)
    axs[1].legend()

    fig.tight_layout()
    return fig


def plot_final_design_operating_points(baseline: DesignEvaluation, redesign: DesignEvaluation):
    airfoil = redesign.solution.geometry.airfoil
    alpha_polar_deg = np.rad2deg(airfoil.alpha)

    baseline_data = spanwise_data(baseline)
    redesign_data = spanwise_data(redesign)

    baseline_alpha_rad = np.deg2rad(baseline_data["alpha"])
    redesign_alpha_rad = np.deg2rad(redesign_data["alpha"])
    baseline_cl = np.array([airfoil.lookup(alpha, 'cl') for alpha in baseline_alpha_rad])
    baseline_cd = np.array([airfoil.lookup(alpha, 'cd') for alpha in baseline_alpha_rad])
    redesign_cl = np.array([airfoil.lookup(alpha, 'cl') for alpha in redesign_alpha_rad])
    redesign_cd = np.array([airfoil.lookup(alpha, 'cd') for alpha in redesign_alpha_rad])

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle('Task h: Airfoil Operating Points for Baseline and Final Redesign')

    axs[0].plot(alpha_polar_deg, airfoil.cl, linewidth=2, color='tab:blue', label='Polar')
    axs[0].scatter(baseline_data["alpha"], baseline_cl, s=20, label='Baseline')
    axs[0].scatter(redesign_data["alpha"], redesign_cl, s=20, label='Final redesign')
    axs[0].set_xlabel('Angle of Attack [deg]')
    axs[0].set_ylabel('Lift Coefficient Cl [-]')
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(alpha_polar_deg, airfoil.cd, linewidth=2, color='tab:red', label='Polar')
    axs[1].scatter(baseline_data["alpha"], baseline_cd, s=20, label='Baseline')
    axs[1].scatter(redesign_data["alpha"], redesign_cd, s=20, label='Final redesign')
    axs[1].set_xlabel('Angle of Attack [deg]')
    axs[1].set_ylabel('Drag Coefficient Cd [-]')
    axs[1].grid(True)
    axs[1].legend()

    fig.tight_layout()
    return fig


def stagnation_pressure_profiles(evaluation: DesignEvaluation) -> dict[str, np.ndarray]:
    solution = evaluation.solution
    data = spanwise_data(evaluation)
    free_stream_velocity = solution.geometry.freeStreamVelocity
    air_density = solution.airDensity
    dynamic_pressure = 0.5 * air_density * free_stream_velocity ** 2
    a = data["a"]

    disk_velocity = free_stream_velocity * (1 - a) # FIX: unused
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


def save_stagnation_pressure_csv(path: Path, evaluation: DesignEvaluation) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pressure = stagnation_pressure_profiles(evaluation)
    with path.open("w", encoding="ascii", newline="") as file:
        file.write(
            "r_over_R,p0_upstream_Pa,p0_before_disk_Pa,p0_after_disk_Pa,p0_far_wake_Pa,"
            "p0_upstream_star,p0_before_disk_star,p0_after_disk_star,p0_far_wake_star\n")
        for index in range(len(pressure["radial"])):
            file.write(
                f"{pressure['radial'][index]:.6f},{pressure['p0_upstream'][index]:.6f},"
                f"{pressure['p0_before_disk'][index]:.6f},{pressure['p0_after_disk'][index]:.6f},"
                f"{pressure['p0_far_wake'][index]:.6f},{pressure['p0_upstream_star'][index]:.6f},"
                f"{pressure['p0_before_disk_star'][index]:.6f},{pressure['p0_after_disk_star'][index]:.6f},"
                f"{pressure['p0_far_wake_star'][index]:.6f}\n")


def plot_stagnation_pressure_distribution(baseline: DesignEvaluation, redesign: DesignEvaluation):
    baseline_pressure = stagnation_pressure_profiles(baseline)
    redesign_pressure = stagnation_pressure_profiles(redesign)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle('Task i: Stagnation Pressure Distribution for the Final Redesign')

    axs[0].plot(redesign_pressure["radial"], redesign_pressure["p0_upstream_star"], linewidth=2, label='1: Far upstream')
    axs[0].plot(redesign_pressure["radial"], redesign_pressure["p0_before_disk_star"], linewidth=2, linestyle='--', label='2: Before disk')
    axs[0].plot(redesign_pressure["radial"], redesign_pressure["p0_after_disk_star"], linewidth=2, label='3: After disk')
    axs[0].plot(redesign_pressure["radial"], redesign_pressure["p0_far_wake_star"], linewidth=2, linestyle='--', label='4: Far wake')
    axs[0].set_xlabel('Radial Position (r/R)')
    axs[0].set_ylabel('Normalized Stagnation Pressure [-]')
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(baseline_pressure["radial"], baseline_pressure["p0_after_disk_star"], linewidth=2, label='Baseline')
    axs[1].plot(redesign_pressure["radial"], redesign_pressure["p0_after_disk_star"], linewidth=2, label='Final redesign')
    axs[1].set_xlabel('Radial Position (r/R)')
    axs[1].set_ylabel('Wake Stagnation Pressure [-]')
    axs[1].grid(True)
    axs[1].legend()

    fig.tight_layout()
    return fig


def write_report_task_notes(task_dir: Path, filename: str, lines: list[str]) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / filename).write_text("\n".join(lines), encoding="ascii")


def generate_task_g_outputs( # FIX: Move to main.py
    baseline: DesignEvaluation,
    stage_a: DesignEvaluation,
    stage_b: DesignEvaluation,
    stage_b_results: list[DesignEvaluation]
) -> None:
    task_dir = REPORT_DIR / "g"
    task_dir.mkdir(parents=True, exist_ok=True)

    save_stage_b_scan(task_dir / "g_design_search.csv", stage_b_results)
    save_design_summary(task_dir / "g_design_summary.csv", [baseline, stage_a, stage_b])
    Plotting.SaveFigure(plot_cp_vs_twist_scale(stage_b_results), task_dir / "g_design_search.png")

    notes = [
        "# Task g notes",
        "",
        "Design approach used for point 2:",
        f"- Fix TSR = {TARGET_TSR:.0f} and target CT = {TARGET_CT:.2f}.",
        "- Stage A changes collective pitch only and solves for the pitch that meets the CT target.",
        "- Stage B scans twist-scale values and, for each candidate, solves collective pitch to meet the same CT target.",
        "- The final design is the candidate with the highest CP among the tested Stage B cases.",
        "",
        f"Selected final design: twist scale = {stage_b.twist_scale:.4f}, pitch = {stage_b.pitch_deg:.3f} deg.",
        f"Baseline CP at TSR 8: {baseline.cp:.4f}.",
        f"Stage A CP at CT = 0.75: {stage_a.cp:.4f}.",
        f"Stage B CP at CT = 0.75: {stage_b.cp:.4f}.",
        "- The gain from Stage B over Stage A is small, so the extra twist freedom gives only a modest improvement for the present parameterization.",
    ]
    write_report_task_notes(task_dir, "g_notes.md", notes)


def generate_task_h_outputs(baseline: DesignEvaluation, stage_a: DesignEvaluation, stage_b: DesignEvaluation) -> None:
    task_dir = REPORT_DIR / "h"
    task_dir.mkdir(parents=True, exist_ok=True)

    save_design_summary(task_dir / "h_design_summary.csv", [baseline, stage_a, stage_b])
    save_spanwise_design_csv(task_dir / "h_baseline_spanwise.csv", baseline)
    save_spanwise_design_csv(task_dir / "h_stage_b_spanwise.csv", stage_b)
    Plotting.SaveFigure(plot_final_design_geometry(baseline, stage_b), task_dir / "h_geometry_comparison.png")
    Plotting.SaveFigure(plot_final_design_performance(baseline, stage_b), task_dir / "h_performance_comparison.png")
    Plotting.SaveFigure(plot_final_design_operating_points(baseline, stage_b), task_dir / "h_operating_points.png")
    Plotting.SaveFigure(plot_design_summary_table(baseline, stage_a, stage_b), task_dir / "h_summary_table.png")

    notes = [
        "# Task h notes",
        "",
        "This folder compares the baseline rotor with the final redesign selected for point 2.",
        "",
        f"- Final redesign pitch: {stage_b.pitch_deg:.3f} deg.",
        f"- Final redesign twist scale: {stage_b.twist_scale:.4f}.",
        f"- Final redesign CT: {stage_b.ct:.4f}.",
        f"- Final redesign CP: {stage_b.cp:.4f}.",
        f"- Baseline CP at TSR 8: {baseline.cp:.4f}.",
        f"- CP increase relative to baseline: {stage_b.cp - baseline.cp:.4f}.",
        "- The present redesign changes collective pitch and twist only; the chord distribution remains the baseline distribution.",
        "- Stage A is included in the summary table as an intermediate design step, while the detailed spanwise comparison focuses on the final Stage B design.",
    ]
    write_report_task_notes(task_dir, "h_notes.md", notes)


def generate_task_i_outputs(baseline: DesignEvaluation, stage_b: DesignEvaluation) -> None:
    task_dir = REPORT_DIR / "i"
    task_dir.mkdir(parents=True, exist_ok=True)

    save_stagnation_pressure_csv(task_dir / "i_stagnation_pressure.csv", stage_b)
    Plotting.SaveFigure(plot_stagnation_pressure_distribution(baseline, stage_b), task_dir / "i_stagnation_pressure_distribution.png")

    notes = [
        "# Task i notes",
        "",
        "The stagnation-pressure plot uses four axial locations from actuator-disk theory:",
        "- 1: far upstream",
        "- 2: just before the disk",
        "- 3: just after the disk",
        "- 4: far wake",
        "",
        "The plotted quantity is normalized stagnation pressure p0* = (p0 - p_inf) / (0.5 rho U_inf^2).",
        "- Locations 1 and 2 coincide because there is no energy extraction before the disk.",
        "- Locations 3 and 4 coincide because the stagnation-pressure loss occurs across the rotor plane and then remains constant downstream in the idealized 1D interpretation.",
        "- The right-hand panel compares the wake stagnation-pressure level of the baseline and final redesign.",
    ]
    write_report_task_notes(task_dir, "i_notes.md", notes)


def run_point_2_report_tasks( # FIX: Move to main
    baseline: DesignEvaluation,
    stage_a: DesignEvaluation,
    stage_b: DesignEvaluation,
    stage_b_results: list[DesignEvaluation]
) -> None:
    generate_task_g_outputs(baseline, stage_a, stage_b, stage_b_results)
    generate_task_h_outputs(baseline, stage_a, stage_b)
    generate_task_i_outputs(baseline, stage_b)


def run_point_2() -> tuple[DesignEvaluation, DesignEvaluation, DesignEvaluation, list[DesignEvaluation]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline, stage_a, stage_b, stage_b_results = compute_point_2_designs()

    save_design_summary(OUTPUT_DIR / "design_summary.csv", [baseline, stage_a, stage_b])
    save_stage_b_scan(OUTPUT_DIR / "stage_b_twist_scale_scan.csv", stage_b_results)
    save_spanwise_design_csv(OUTPUT_DIR / "baseline_tsr8_spanwise.csv", baseline)
    save_spanwise_design_csv(OUTPUT_DIR / "stage_a_spanwise.csv", stage_a)
    save_spanwise_design_csv(OUTPUT_DIR / "stage_b_spanwise.csv", stage_b)

    Plotting.SaveFigure(plot_cp_vs_twist_scale(stage_b_results), OUTPUT_DIR / "stage_b_twist_scale_scan.png")
    Plotting.SaveFigure(plot_design_comparison(baseline, stage_a, stage_b), OUTPUT_DIR / "baseline_stageA_stageB_comparison.png")
    Plotting.SaveFigure(plot_design_distributions(baseline, stage_a, stage_b), OUTPUT_DIR / "design_distributions.png")
    Plotting.SaveFigure(plot_design_summary_table(baseline, stage_a, stage_b), OUTPUT_DIR / "design_summary_table.png")
    Plotting.SaveFigure(Plotting.PlotSpanwiseResults(stage_a.solution), OUTPUT_DIR / "stage_a_spanwise.png")
    Plotting.SaveFigure(Plotting.PlotSpanwiseResults(stage_b.solution), OUTPUT_DIR / "stage_b_spanwise.png")

    write_notes(OUTPUT_DIR / "point_2_notes.md", baseline, stage_a, stage_b)

    print("\nPoint 2 design summary")
    print(f"{'Case':<12} {'Pitch [deg]':>12} {'Twist Scale':>12} {'CT [-]':>10} {'CP [-]':>10}")
    for evaluation in [baseline, stage_a, stage_b]:
        print(
            f"{evaluation.label:<12} "
            f"{evaluation.pitch_deg:12.4f} "
            f"{evaluation.twist_scale:12.4f} "
            f"{evaluation.ct:10.4f} "
            f"{evaluation.cp:10.4f}"
        )
    print(f"{'actuator_disk':<12} {'-':>12} {'-':>12} {TARGET_CT:10.4f} {actuator_disk_cp(TARGET_CT):10.4f}")
    print(f"\nSaved point-2 results to: {OUTPUT_DIR.resolve()}")
    return baseline, stage_a, stage_b, stage_b_results


if __name__ == "__main__":
    point_2_bundle = run_point_2()
    run_point_2_report_tasks(*point_2_bundle)
