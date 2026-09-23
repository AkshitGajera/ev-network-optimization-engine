"""
Visualization Module for EV Infrastructure & Fleet Routing.
Generates:
1. Interactive geospatial Folium maps (Stations, Coverage Radii, Fleet Tours).
2. Professional Pareto frontier & solver benchmark figures.
"""

from typing import List, Optional
import folium
from folium import plugins
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.data_processing import OptimizationDataset
from src.facility_location import OptimizationResult
from src.ev_routing import FleetTour


def create_interactive_map(
    dataset: OptimizationDataset,
    opt_result: OptimizationResult,
    fleet_tours: Optional[List[FleetTour]] = None,
    coverage_radius_km: float = 15.0,
    output_html_path: str = "optimized_network_map.html",
) -> folium.Map:
    """Builds an interactive multi-layer Leaflet/Folium map with:

    - Demand Points (Heatmap & Circles)
    - Optimal Selected Stations with coverage radius circles
    - Candidate unselected stations
    - EV Fleet Delivery Routes with battery SoC annotations
    """
    center_lat = float(np.mean(dataset.demand_points[:, 0]))
    center_lon = float(np.mean(dataset.demand_points[:, 1]))

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=8,
        tiles="cartodbpositron",
    )

    # 1. Demand Layer
    demand_fg = folium.FeatureGroup(name="Demand Clusters (kW)", show=True)
    for i in range(len(dataset.demand_points)):
        lat, lon = dataset.demand_points[i]
        weight = dataset.demand_weights[i]
        is_covered = i in opt_result.covered_demand_indices
        color = "#2b8a3e" if is_covered else "#c92a2a"

        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color=color,
            fill=True,
            fill_opacity=0.6,
            popup=f"Demand Node #{i}<br>Load: {weight:.1f} kW<br>Status: {'Covered' if is_covered else 'Uncovered'}",
        ).add_to(demand_fg)
    demand_fg.add_to(m)

    # 2. Selected Optimal Charging Stations
    stations_fg = folium.FeatureGroup(name="Optimal Stations (MILP)", show=True)
    for idx in opt_result.selected_indices:
        lat, lon = dataset.candidate_locations[idx]
        cost = dataset.candidate_costs[idx]
        cap = dataset.candidate_capacities[idx]

        folium.Marker(
            location=[lat, lon],
            icon=folium.Icon(color="blue", icon="bolt", prefix="fa"),
            popup=f"<b>Selected Station #{idx}</b><br>CapEx: ${cost:,.0f}<br>Capacity: {cap:.0f} kW",
        ).add_to(stations_fg)

        # Coverage buffer circle
        folium.Circle(
            location=[lat, lon],
            radius=coverage_radius_km * 1000.0,
            color="#1c7ed6",
            fill=True,
            fill_opacity=0.12,
            weight=1.5,
        ).add_to(stations_fg)
    stations_fg.add_to(m)

    # 3. EV Fleet Routing Tours
    if fleet_tours:
        routes_fg = folium.FeatureGroup(name="EV Fleet Delivery Tours", show=True)
        colors = ["#f03e3e", "#7950f2", "#12b886", "#e67700", "#4263eb"]

        for t_idx, tour in enumerate(fleet_tours):
            color = colors[t_idx % len(colors)]
            coords = [[s.lat, s.lon] for s in tour.stops]

            # Route polyline
            folium.PolyLine(
                locations=coords,
                color=color,
                weight=3.5,
                opacity=0.8,
                dash_array="6" if not tour.is_feasible else None,
                popup=f"Vehicle #{tour.vehicle_id}<br>Distance: {tour.total_distance_km} km<br>Charges: {tour.charging_stops_count}",
            ).add_to(routes_fg)

            # Route waypoint markers
            for stop in tour.stops:
                if stop.node_type == "depot":
                    folium.Marker(
                        location=[stop.lat, stop.lon],
                        icon=folium.Icon(color="black", icon="home", prefix="fa"),
                        popup=f"Fleet Base Depot<br>SoC: {stop.departure_soc_pct:.0f}%",
                    ).add_to(routes_fg)
                elif stop.node_type == "charging_station":
                    folium.CircleMarker(
                        location=[stop.lat, stop.lon],
                        radius=7,
                        color="#d9480f",
                        fill=True,
                        fill_color="#ffd43b",
                        fill_opacity=1.0,
                        popup=f"En-route Recharging Stop<br>Arrival SoC: {stop.arrival_soc_pct:.1f}%<br>Departure SoC: {stop.departure_soc_pct:.1f}%",
                    ).add_to(routes_fg)

        routes_fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(output_html_path)
    return m


def plot_pareto_benchmark(df_pareto: pd.DataFrame, save_path: str = "pareto_benchmark.png"):
    """Generates publication-quality dual-axis chart:

    - Investment vs Coverage (%)
    - Solver Runtime (MILP vs Greedy)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)

    # Subplot 1: Pareto Efficiency Curve
    ax1.plot(
        df_pareto["budget"] / 1000.0,
        df_pareto["milp_coverage_pct"],
        "o-",
        color="#1c7ed6",
        linewidth=2.5,
        label="Exact MILP (Optimal)",
    )
    ax1.plot(
        df_pareto["budget"] / 1000.0,
        df_pareto["greedy_coverage_pct"],
        "s--",
        color="#e67700",
        linewidth=2.0,
        label="Greedy Heuristic",
    )
    ax1.set_xlabel("Capital Budget ($k)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Demand Coverage (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Pareto Frontier: Investment vs. Demand Coverage", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", frameon=True)

    # Subplot 2: Computational Runtime Scalability
    ax2.plot(
        df_pareto["budget"] / 1000.0,
        df_pareto["milp_time_ms"],
        "o-",
        color="#c92a2a",
        linewidth=2.0,
        label="Exact MILP (ms)",
    )
    ax2.plot(
        df_pareto["budget"] / 1000.0,
        df_pareto["greedy_time_ms"],
        "s-",
        color="#2b8a3e",
        linewidth=2.0,
        label="Greedy Heuristic (ms)",
    )
    ax2.set_xlabel("Capital Budget ($k)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Solve Time (milliseconds)", fontsize=11, fontweight="bold")
    ax2.set_title("Solver Runtime Scalability Benchmark", fontsize=12, fontweight="bold")
    ax2.set_yscale("log")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
