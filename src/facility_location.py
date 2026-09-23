"""
Mathematical Optimization Solvers for Facility Location & Station Placement.
Implements:
1. Exact Mixed-Integer Linear Programming (MILP)
2. Greedy Marginal-Gain Heuristic
3. LP Relaxation for Theoretical Upper Bound & Optimality Gap Computation
"""

from dataclasses import dataclass
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import LinearConstraint, milp
from scipy.sparse import csc_matrix, lil_matrix
from src.data_processing import OptimizationDataset


@dataclass
class OptimizationResult:
    solver_name: str
    selected_indices: List[int]
    covered_demand_indices: List[int]
    total_covered_demand: float
    total_demand: float
    coverage_percentage: float
    total_cost: float
    budget: float
    solve_time_sec: float
    optimality_gap_percent: Optional[float] = None
    status: str = "optimal"


class FacilityLocationSolver:
    """Exact and Heuristic solvers for the Budgeted Maximal Covering Location Problem (MCLP)."""

    def __init__(self, dataset: OptimizationDataset, coverage_radius_km: float = 15.0):
        self.dataset = dataset
        self.R = coverage_radius_km
        self.num_demand = len(dataset.demand_points)
        self.num_candidates = len(dataset.candidate_locations)

        # Coverage adjacency matrix: A[i, j] = 1 if distance(i, j) <= R
        self.coverage_matrix = (dataset.distance_matrix <= self.R).astype(int)

    def solve_milp(self, budget: float, max_stations: Optional[int] = None) -> OptimizationResult:
        """Exact Mixed-Integer Linear Program (MILP) formulation:

        Maximize: sum_i (w_i * y_i)
        Subject to:
          y_i - sum_{j in N_i} x_j <= 0  (demand i only covered if at least one adjacent station is built)
          sum_j (cost_j * x_j) <= Budget (budget constraint)
          sum_j (x_j) <= max_stations    (optional station count limit)
          x_j in {0, 1}, y_i in {0, 1}
        """
        start_time = time.perf_counter()

        n_x = self.num_candidates
        n_y = self.num_demand
        n_vars = n_x + n_y

        # Variable vector layout: [x_0, ..., x_{m-1}, y_0, ..., y_{n-1}]
        # SciPy minimizes c^T * z, so minimize -sum(w_i * y_i)
        c = np.zeros(n_vars)
        c[n_x:] = -self.dataset.demand_weights

        # All variables are binary integers
        integrality = np.ones(n_vars)

        # Variable bounds: x_j in [0, 1], y_i in [0, 1]
        lb = np.zeros(n_vars)
        ub = np.ones(n_vars)

        # Constraints
        # 1. Coverage constraints: y_i - sum_{j: A[i,j]=1} x_j <= 0
        # -> -sum(x_j) + y_i <= 0, lower bound = -inf, upper bound = 0
        A_cov = lil_matrix((n_y, n_vars))
        for i in range(n_y):
            covering_candidates = np.where(self.coverage_matrix[i, :] == 1)[0]
            for j in covering_candidates:
                A_cov[i, j] = -1.0
            A_cov[i, n_x + i] = 1.0

        lhs_cov = np.full(n_y, -np.inf)
        rhs_cov = np.zeros(n_y)

        # 2. Budget constraint: sum_j (cost_j * x_j) <= budget
        A_budget = lil_matrix((1, n_vars))
        for j in range(n_x):
            A_budget[0, j] = self.dataset.candidate_costs[j]
        lhs_budget = np.array([-np.inf])
        rhs_budget = np.array([budget])

        # Stack constraints
        rows = [A_cov, A_budget]
        lhs_list = [lhs_cov, lhs_budget]
        rhs_list = [rhs_cov, rhs_budget]

        if max_stations is not None:
            A_max = lil_matrix((1, n_vars))
            A_max[0, :n_x] = 1.0
            rows.append(A_max)
            lhs_list.append(np.array([-np.inf]))
            rhs_list.append(np.array([float(max_stations)]))

        from scipy.sparse import vstack
        A_all = csc_matrix(vstack(rows))
        lhs_all = np.concatenate(lhs_list)
        rhs_all = np.concatenate(rhs_list)

        constraints = LinearConstraint(A_all, lhs_all, rhs_all)

        res = milp(
            c=c,
            integrality=integrality,
            bounds=(lb, ub),
            constraints=constraints,
        )

        solve_time = time.perf_counter() - start_time

        if res.success:
            sol = res.x
            selected = [j for j in range(n_x) if sol[j] > 0.5]
            covered = [i for i in range(n_y) if sol[n_x + i] > 0.5]
            covered_demand = float(np.sum(self.dataset.demand_weights[covered]))
            total_demand = float(np.sum(self.dataset.demand_weights))
            cost = float(np.sum(self.dataset.candidate_costs[selected]))
            cov_pct = (covered_demand / total_demand * 100.0) if total_demand > 0 else 0.0

            # Solve LP relaxation for optimality gap
            lp_bound = self._solve_lp_relaxation(budget, max_stations)
            gap = (
                max(0.0, ((lp_bound - covered_demand) / lp_bound) * 100.0)
                if lp_bound > 0
                else 0.0
            )

            return OptimizationResult(
                solver_name="Exact MILP (SciPy Highs)",
                selected_indices=selected,
                covered_demand_indices=covered,
                total_covered_demand=covered_demand,
                total_demand=total_demand,
                coverage_percentage=cov_pct,
                total_cost=cost,
                budget=budget,
                solve_time_sec=solve_time,
                optimality_gap_percent=round(gap, 2),
                status="optimal",
            )
        else:
            return OptimizationResult(
                solver_name="Exact MILP (SciPy Highs)",
                selected_indices=[],
                covered_demand_indices=[],
                total_covered_demand=0.0,
                total_demand=float(np.sum(self.dataset.demand_weights)),
                coverage_percentage=0.0,
                total_cost=0.0,
                budget=budget,
                solve_time_sec=solve_time,
                optimality_gap_percent=None,
                status=f"infeasible: {res.status}",
            )

    def _solve_lp_relaxation(self, budget: float, max_stations: Optional[int] = None) -> float:
        """Solves continuous LP relaxation (integrality=0) to establish upper bound."""
        n_x = self.num_candidates
        n_y = self.num_demand
        n_vars = n_x + n_y

        c = np.zeros(n_vars)
        c[n_x:] = -self.dataset.demand_weights
        integrality = np.zeros(n_vars)  # Continuous variables

        lb = np.zeros(n_vars)
        ub = np.ones(n_vars)

        A_cov = lil_matrix((n_y, n_vars))
        for i in range(n_y):
            covering_candidates = np.where(self.coverage_matrix[i, :] == 1)[0]
            for j in covering_candidates:
                A_cov[i, j] = -1.0
            A_cov[i, n_x + i] = 1.0

        A_budget = lil_matrix((1, n_vars))
        for j in range(n_x):
            A_budget[0, j] = self.dataset.candidate_costs[j]

        from scipy.sparse import vstack
        rows = [A_cov, A_budget]
        lhs_list = [np.full(n_y, -np.inf), np.array([-np.inf])]
        rhs_list = [np.zeros(n_y), np.array([budget])]

        if max_stations is not None:
            A_max = lil_matrix((1, n_vars))
            A_max[0, :n_x] = 1.0
            rows.append(A_max)
            lhs_list.append(np.array([-np.inf]))
            rhs_list.append(np.array([float(max_stations)]))

        A_all = csc_matrix(vstack(rows))
        constraints = LinearConstraint(A_all, np.concatenate(lhs_list), np.concatenate(rhs_list))

        res = milp(c=c, integrality=integrality, bounds=(lb, ub), constraints=constraints)
        if res.success:
            return float(-res.fun)
        return float(np.sum(self.dataset.demand_weights))

    def solve_greedy(self, budget: float, max_stations: Optional[int] = None) -> OptimizationResult:
        """Greedy Heuristic with Marginal Gain per Cost (Benefit-to-Cost Ratio):

        At each iteration, select candidate j* = argmax (Delta_Covered_Demand / Cost_j).
        Fast O(M * N) complexity suitable for real-time dispatch and large-scale networks.
        """
        start_time = time.perf_counter()

        selected = []
        covered_mask = np.zeros(self.num_demand, dtype=bool)
        remaining_budget = budget

        while True:
            best_j = -1
            best_ratio = -1.0
            best_gain = 0.0

            if max_stations is not None and len(selected) >= max_stations:
                break

            for j in range(self.num_candidates):
                if j in selected:
                    continue

                cost = self.dataset.candidate_costs[j]
                if cost > remaining_budget:
                    continue

                # Find newly covered demand nodes
                candidate_coverage = self.coverage_matrix[:, j] == 1
                newly_covered = candidate_coverage & (~covered_mask)
                gain = np.sum(self.dataset.demand_weights[newly_covered])

                ratio = gain / (cost + 1e-6)
                if ratio > best_ratio and gain > 0:
                    best_ratio = ratio
                    best_j = j
                    best_gain = gain

            if best_j == -1:
                break

            selected.append(best_j)
            remaining_budget -= self.dataset.candidate_costs[best_j]
            covered_mask |= (self.coverage_matrix[:, best_j] == 1)

        solve_time = time.perf_counter() - start_time
        covered_indices = np.where(covered_mask)[0].tolist()
        total_demand = float(np.sum(self.dataset.demand_weights))
        covered_demand = float(np.sum(self.dataset.demand_weights[covered_indices]))
        total_cost = float(np.sum(self.dataset.candidate_costs[selected]))
        cov_pct = (covered_demand / total_demand * 100.0) if total_demand > 0 else 0.0

        return OptimizationResult(
            solver_name="Greedy Benefit-Cost Heuristic",
            selected_indices=selected,
            covered_demand_indices=covered_indices,
            total_covered_demand=covered_demand,
            total_demand=total_demand,
            coverage_percentage=cov_pct,
            total_cost=total_cost,
            budget=budget,
            solve_time_sec=solve_time,
            optimality_gap_percent=None,
            status="heuristic_completed",
        )
