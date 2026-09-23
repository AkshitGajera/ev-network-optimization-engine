"""
Pareto Frontier & Sensitivity Analysis Module.
Explores the investment trade-off curve (CapEx vs. Service Level) and solver efficiency.
"""

from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from src.facility_location import FacilityLocationSolver, OptimizationResult
from src.data_processing import OptimizationDataset


class ParetoAnalyzer:
    """Computes Pareto frontier curves and benchmark metrics across varying capital budgets."""

    def __init__(self, solver: FacilityLocationSolver):
        self.solver = solver

    def run_budget_sweep(
        self,
        min_budget: float = 50000.0,
        max_budget: float = 600000.0,
        num_steps: int = 10,
        max_stations: int = None,
    ) -> pd.DataFrame:
        """Sweeps budget values to generate the Pareto Efficient Investment Frontier

        comparing Exact MILP with Greedy Heuristics.
        """
        budget_range = np.linspace(min_budget, max_budget, num_steps)
        records = []

        for b in budget_range:
            # 1. Solve with MILP
            res_milp = self.solver.solve_milp(budget=b, max_stations=max_stations)
            # 2. Solve with Greedy
            res_greedy = self.solver.solve_greedy(budget=b, max_stations=max_stations)

            records.append({
                "budget": b,
                "milp_cost": res_milp.total_cost,
                "milp_covered_demand_kw": res_milp.total_covered_demand,
                "milp_coverage_pct": res_milp.coverage_percentage,
                "milp_num_stations": len(res_milp.selected_indices),
                "milp_time_ms": res_milp.solve_time_sec * 1000.0,
                "milp_optimality_gap_pct": res_milp.optimality_gap_percent or 0.0,
                "greedy_cost": res_greedy.total_cost,
                "greedy_covered_demand_kw": res_greedy.total_covered_demand,
                "greedy_coverage_pct": res_greedy.coverage_percentage,
                "greedy_num_stations": len(res_greedy.selected_indices),
                "greedy_time_ms": res_greedy.solve_time_sec * 1000.0,
                "heuristic_efficiency_ratio": (
                    res_greedy.total_covered_demand / (res_milp.total_covered_demand + 1e-6)
                ) * 100.0,
            })

        df_pareto = pd.DataFrame(records)
        return df_pareto

    def find_elbow_budget(self, df_pareto: pd.DataFrame) -> Dict[str, float]:
        """Detects the point of diminishing returns (knee/elbow) in the investment curve.

        Calculates marginal coverage gain per dollar.
        """
        marginal_gains = []
        for i in range(1, len(df_pareto)):
            d_cov = df_pareto.loc[i, "milp_coverage_pct"] - df_pareto.loc[i - 1, "milp_coverage_pct"]
            d_cost = df_pareto.loc[i, "milp_cost"] - df_pareto.loc[i - 1, "milp_cost"]
            roi = d_cov / (d_cost + 1e-6) * 10000.0  # % gain per $10k
            marginal_gains.append(roi)

        # The elbow is typically where the drop in marginal gain is steepest
        diffs = np.diff(marginal_gains)
        elbow_idx = int(np.argmin(diffs)) + 1 if len(diffs) > 0 else 0

        recommended = df_pareto.iloc[elbow_idx]
        return {
            "recommended_budget": float(recommended["budget"]),
            "expected_coverage_pct": float(recommended["milp_coverage_pct"]),
            "stations_built": int(recommended["milp_num_stations"]),
            "actual_cost": float(recommended["milp_cost"]),
        }
