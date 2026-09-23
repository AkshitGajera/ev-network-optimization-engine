"""
Headless Optimization Benchmark & Pipeline Runner.
Runs:
1. Data loading & spatial preprocessing
2. Exact MILP vs Greedy benchmark
3. Pareto Frontier sensitivity analysis
4. Electric Vehicle Routing Problem (EVRP) tour planning with battery dynamics
5. Generation of interactive maps (HTML) and benchmark charts (PNG)
"""

import os
import sys
import numpy as np
import pandas as pd

from src.data_processing import (
    load_and_preprocess_stations,
    prepare_optimization_instance,
)
from src.facility_location import FacilityLocationSolver
from src.pareto_analysis import ParetoAnalyzer
from src.ev_routing import ElectricVehicleRouter
from src.visualizer import create_interactive_map, plot_pareto_benchmark


def main():
    print("=" * 70)
    print("  EV INFRASTRUCTURE & FLEET ROUTING OPTIMIZATION ENGINE")
    print("=" * 70)

    # 1. Load Data
    data_path = "Indian_EV_Stations_Simplified.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        sys.exit(1)

    print(f"\n[1/5] Loading and cleaning dataset from {data_path}...")
    df_clean = load_and_preprocess_stations(data_path)
    print(f"      -> Cleaned records in operational corridor: {len(df_clean)} stations")

    # 2. Build Optimization Instance
    print("\n[2/5] Synthesizing Spatial Optimization Problem Instance...")
    dataset = prepare_optimization_instance(
        df_clean,
        grid_size=36,               # 36 candidate station sites
        sample_demand_nodes=120,    # 120 demand clusters
        random_seed=42,
    )
    print(f"      -> Demand nodes: {len(dataset.demand_points)} | Total Demand: {np.sum(dataset.demand_weights):,.1f} kW")
    print(f"      -> Candidate sites: {len(dataset.candidate_locations)} | Total Potential CapEx: ${np.sum(dataset.candidate_costs):,.0f}")

    # 3. Solver Setup & Benchmark
    coverage_radius = 18.0  # km
    solver = FacilityLocationSolver(dataset, coverage_radius_km=coverage_radius)
    test_budget = 200000.0  # $200k budget

    print(f"\n[3/5] Solving Strategic Station Placement (Budget: ${test_budget:,.0f}, Radius: {coverage_radius} km)...")
    res_milp = solver.solve_milp(budget=test_budget)
    res_greedy = solver.solve_greedy(budget=test_budget)

    print("\n" + "-" * 70)
    print(f"  {'Metric':<28} | {'Exact MILP':<18} | {'Greedy Heuristic':<18}")
    print("-" * 70)
    print(f"  {'Coverage Percentage':<28} | {res_milp.coverage_percentage:>16.2f}% | {res_greedy.coverage_percentage:>16.2f}%")
    print(f"  {'Covered Demand (kW)':<28} | {res_milp.total_covered_demand:>16.1f} | {res_greedy.total_covered_demand:>16.1f}")
    print(f"  {'Stations Built':<28} | {len(res_milp.selected_indices):>18} | {len(res_greedy.selected_indices):>18}")
    print(f"  {'Total Capital Spent':<28} | ${res_milp.total_cost:>16,.0f} | ${res_greedy.total_cost:>16,.0f}")
    print(f"  {'Solve Time (ms)':<28} | {res_milp.solve_time_sec*1000:>16.2f} | {res_greedy.solve_time_sec*1000:>16.2f}")
    print(f"  {'Optimality Gap (%)':<28} | {res_milp.optimality_gap_percent:>16.2f}% | {'N/A (Heuristic)':>18}")
    print("-" * 70)

    # 4. Pareto Frontier Analysis
    print("\n[4/5] Executing Pareto Frontier Sweep across Budgets ($50k - $500k)...")
    analyzer = ParetoAnalyzer(solver)
    df_pareto = analyzer.run_budget_sweep(min_budget=50000.0, max_budget=500000.0, num_steps=10)
    df_pareto.to_csv("pareto_benchmark_results.csv", index=False)
    plot_pareto_benchmark(df_pareto, save_path="pareto_benchmark.png")
    print("      -> Saved Pareto metrics to 'pareto_benchmark_results.csv'")
    print("      -> Saved dual-axis benchmark plot to 'pareto_benchmark.png'")

    elbow = analyzer.find_elbow_budget(df_pareto)
    print(f"      -> Recommended Optimal Budget (Elbow Point): ${elbow['recommended_budget']:,.0f}")
    print(f"      -> Expected Service Level: {elbow['expected_coverage_pct']:.1f}% with {elbow['stations_built']} stations")

    # 5. Operational EV Routing Simulation
    print("\n[5/5] Simulating Operational Fleet Routing (EVRP) with Battery SoC...")
    selected_stations_coords = dataset.candidate_locations[res_milp.selected_indices]

    # Select delivery customers in the central service zone
    depot_coord = (float(np.mean(dataset.demand_points[:10, 0])), float(np.mean(dataset.demand_points[:10, 1])))
    customers_subset = dataset.demand_points[10:35]  # 25 delivery stops

    router = ElectricVehicleRouter(
        depot_coords=depot_coord,
        customer_coords=customers_subset,
        station_coords=selected_stations_coords,
        battery_capacity_kwh=60.0,
        consumption_kwh_per_km=0.25,
        min_safe_soc_pct=15.0,
        target_charge_soc_pct=85.0,
    )

    tours = router.plan_fleet_routes(num_vehicles=3)
    print("\n  Fleet Routing Tour Summary:")
    tour_records = []
    for tour in tours:
        print(f"    - Vehicle #{tour.vehicle_id}: Distance = {tour.total_distance_km:.1f} km, "
              f"En-route Recharging Stops = {tour.charging_stops_count}, "
              f"Feasible = {tour.is_feasible}")
        tour_records.append({
            "vehicle_id": tour.vehicle_id,
            "distance_km": tour.total_distance_km,
            "recharge_stops": tour.charging_stops_count,
            "is_feasible": tour.is_feasible,
            "total_stops": len(tour.stops),
        })
    pd.DataFrame(tour_records).to_csv("fleet_tours_summary.csv", index=False)

    # 6. Interactive Geospatial Visualization
    map_file = "optimized_network_map.html"
    create_interactive_map(
        dataset=dataset,
        opt_result=res_milp,
        fleet_tours=tours,
        coverage_radius_km=coverage_radius,
        output_html_path=map_file,
    )
    print(f"\n[DONE] Interactive Folium GIS Map exported to '{map_file}'")
    print("=" * 70)
    print("OPTIMIZATION PIPELINE COMPLETED SUCCESSFULLY.")
    print("=" * 70)


if __name__ == "__main__":
    main()
