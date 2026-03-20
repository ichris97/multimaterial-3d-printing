"""
Analysis module: mechanical property prediction, thermal stress, warping, and optimization.
"""
from .mechanical import (
    analyze_layup,
    compute_abd_matrix,
    compute_layer_stiffness_matrix,
    compute_orthotropic_stiffness,
    rotate_stiffness_matrix,
    compute_hashin_shtrikman_bounds,
    compute_interlaminar_shear,
    compute_tsai_wu_failure,
)
from .thermal import thermal_stress_analysis, predict_warping
from .optimizer import optimize_material_distribution, generate_gradient_transition
from .print_estimator import estimate_print_time, estimate_cost
