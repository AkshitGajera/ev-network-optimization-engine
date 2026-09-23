"""
Interactive Streamlit Dashboard: EV Infrastructure & Fleet Optimization Platform.
Run locally with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt

from src.data_processing import (
    load_and_preprocess_stations,
    prepare_optimization_instance,
)
from src.facility_location import FacilityLocationSolver
from src.pareto_analysis import ParetoAnalyzer
from src.ev_routing import ElectricVehicleRouter
from src.visualizer import create_interactive_map

st.set_page_config(
    page_title="EV Infrastructure & Fleet Optimization",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ EV Infrastructure Planning & Fleet Routing Engine")
st.markdown(
    """
    **Dual-Tier Operations Research Platform**:
    1. **Strategic Tier**: Capacitated & Budgeted Facility Location (MILP).
    2. **Operational Tier**: Commercial Electric Vehicle Routing Problem (EVRP) with battery State of Charge (SoC).
    """
)

# Sidebar Controls
st.sidebar.header("🔧 Optimization Parameters")

budget = st.sidebar.slider(
    "Total Capital Budget ($)",
    min_value=50000,
    max_value=600000,
    value=250000,
    step=25000,
    format="$%d",
)

radius = st.sidebar.slider(
    "Coverage Radius (km)",
    min_value=5.0,
    max_value=30.0,
    value=18.0,
    step=1.0,
)

solver_choice = st.sidebar.radio(
    "Mathematical Solver",
    options=["Exact MILP (SciPy Highs)", "Greedy Marginal Benefit-Cost Heuristic"],
)

num_evs = st.sidebar.slider(
    "Operational EV Fleet Size",
    min_value=1,
    max_value=5,
    value=3,
)

battery_kwh = st.sidebar.slider(
    "EV Battery Capacity (kWh)",
    min_value=40.0,
    max_value=100.0,
    value=65.0,
    step=5.0,
)

# Cache data loading
@st.cache_data
def get_cached_dataset():
    df = load_and_preprocess_stations("Indian_EV_Stations_Simplified.csv")
    return df

try:
    df_stations = get_cached_dataset()
    dataset = prepare_optimization_instance(df_stations, grid_size=36, sample_demand_nodes=120)
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Initialize Solver
solver = FacilityLocationSolver(dataset, coverage_radius_km=radius)

if "Exact" in solver_choice:
    result = solver.solve_milp(budget=float(budget))
else:
    result = solver.solve_greedy(budget=float(budget))

# Top KPI Metrics
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Coverage Level", f"{result.coverage_percentage:.1f}%")
kpi2.metric("Demand Covered", f"{result.total_covered_demand:,.0f} kW")
kpi3.metric("Stations Selected", f"{len(result.selected_indices)} / {len(dataset.candidate_locations)}")
kpi4.metric("Capital Invested", f"${result.total_cost:,.0f}")
kpi5.metric("Solver Latency", f"{result.solve_time_sec*1000:.1f} ms")

tab1, tab2, tab3 = st.tabs(["🗺️ Geospatial Optimization Map", "📈 Pareto Frontier Analysis", "🚚 Fleet Routing & Battery SoC"])

with tab1:
    st.subheader("Optimal Station Placement & Coverage Buffers")
    with st.spinner("Generating interactive GIS map..."):
        # Selected stations
        selected_coords = dataset.candidate_locations[result.selected_indices]
        depot_coord = (float(np.mean(dataset.demand_points[:5, 0])), float(np.mean(dataset.demand_points[:5, 1])))
        customers_sample = dataset.demand_points[10:30]

        router = ElectricVehicleRouter(
            depot_coords=depot_coord,
            customer_coords=customers_sample,
            station_coords=selected_coords,
            battery_capacity_kwh=battery_kwh,
        )
        fleet_tours = router.plan_fleet_routes(num_vehicles=num_evs)

        map_obj = create_interactive_map(
            dataset=dataset,
            opt_result=result,
            fleet_tours=fleet_tours,
            coverage_radius_km=radius,
            output_html_path="temp_map.html",
        )
        st_folium(map_obj, width="100%", height=550)

with tab2:
    st.subheader("Budget Sensitivity & Pareto Efficient Trade-Off Curve")
    if st.button("Compute Full Pareto Curve (10 Budget Steps)"):
        with st.spinner("Solving multi-budget MILP frontier..."):
            analyzer = ParetoAnalyzer(solver)
            df_pareto = analyzer.run_budget_sweep(min_budget=50000.0, max_budget=600000.0, num_steps=10)
            elbow = analyzer.find_elbow_budget(df_pareto)

            st.success(
                f"🎯 Recommended Investment (Elbow Point): **${elbow['recommended_budget']:,.0f}** "
                f"achieves **{elbow['expected_coverage_pct']:.1f}%** coverage with **{elbow['stations_built']}** stations."
            )

            fig, ax = plt.subplots(figsize=(10, 4.5))
            ax.plot(df_pareto["budget"] / 1000.0, df_pareto["milp_coverage_pct"], "o-", label="Exact MILP", color="#1c7ed6", lw=2.5)
            ax.plot(df_pareto["budget"] / 1000.0, df_pareto["greedy_coverage_pct"], "s--", label="Greedy Heuristic", color="#e67700", lw=2)
            ax.axvline(budget / 1000.0, color="red", linestyle=":", label=f"Current Budget (${budget/1000:.0f}k)")
            ax.set_xlabel("Capital Budget ($k)")
            ax.set_ylabel("Demand Coverage (%)")
            ax.set_title("Pareto Efficient Frontier: Capital Expenditure vs Service Level")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend()
            st.pyplot(fig)

            st.dataframe(df_pareto.round(2))

with tab3:
    st.subheader("Commercial Fleet Routing (EVRP) Schedule")
    for tour in fleet_tours:
        with st.expander(f"🚛 Vehicle #{tour.vehicle_id} - Total Distance: {tour.total_distance_km:.1f} km (Recharge Stops: {tour.charging_stops_count})"):
            stops_data = [
                {
                    "Stop #": idx,
                    "Type": s.node_type.replace("_", " ").title(),
                    "Arrival SoC": f"{s.arrival_soc_pct:.1f}%",
                    "Departure SoC": f"{s.departure_soc_pct:.1f}%",
                    "Leg Distance": f"{s.distance_from_prev_km:.1f} km",
                }
                for idx, s in enumerate(tour.stops)
            ]
            st.table(pd.DataFrame(stops_data))
