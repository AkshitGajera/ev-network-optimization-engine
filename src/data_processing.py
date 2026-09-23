"""
Data Processing & Geospatial Utilities for EV Infrastructure Planning.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class OptimizationDataset:
    demand_points: np.ndarray  # Shape: (N, 2) -> [lat, lon]
    demand_weights: np.ndarray  # Shape: (N,) -> Demand intensity (kW or vehicle trips)
    candidate_locations: np.ndarray  # Shape: (M, 2) -> [lat, lon]
    candidate_costs: np.ndarray  # Shape: (M,) -> Capital expenditure / installation cost
    candidate_capacities: np.ndarray  # Shape: (M,) -> Maximum power capacity (kW)
    distance_matrix: np.ndarray  # Shape: (N, M) -> Distance in km


def haversine_distance_matrix(
    coords1: np.ndarray, coords2: np.ndarray
) -> np.ndarray:
    """Vectorized calculation of pairwise Haversine distances (in kilometers)

    between two arrays of coordinates [[lat, lon], ...].
    """
    R = 6371.0  # Earth's radius in km

    lat1 = np.radians(coords1[:, 0])[:, np.newaxis]
    lon1 = np.radians(coords1[:, 1])[:, np.newaxis]
    lat2 = np.radians(coords2[:, 0])[np.newaxis, :]
    lon2 = np.radians(coords2[:, 1])[np.newaxis, :]

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.clip(np.sqrt(a), 0, 1.0))
    return R * c


def load_and_preprocess_stations(
    filepath: str = "Indian_EV_Stations_Simplified.csv",
) -> pd.DataFrame:
    """Load and clean EV station dataset."""
    df = pd.read_csv(filepath)

    # Clean coordinates
    df = df.dropna(subset=["Latitude", "Longitude"])
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

    # Clean Power
    df["Power (kW)"] = pd.to_numeric(df["Power (kW)"], errors="coerce")
    median_power = df["Power (kW)"].median()
    df["Power (kW)"] = df["Power (kW)"].fillna(median_power)

    # Focus on primary operational cluster (e.g. Kerala region: Lat 8-13, Lon 74-78)
    df_clean = df[
        (df["Latitude"].between(8.0, 13.0))
        & (df["Longitude"].between(74.5, 78.0))
    ].copy()
    df_clean.reset_index(drop=True, inplace=True)
    return df_clean


def prepare_optimization_instance(
    df_stations: pd.DataFrame,
    grid_size: int = 25,
    sample_demand_nodes: int = 150,
    random_seed: int = 42,
) -> OptimizationDataset:
    """Creates a structured optimization instance containing:

    - Demand points (aggregated spatial clusters & station demand)
    - Candidate station sites (strategic grid + transit junctions)
    - Heterogeneous installation costs and power capacities
    """
    np.random.seed(random_seed)

    # 1. Demand Points
    if len(df_stations) > sample_demand_nodes:
        sample_df = df_stations.sample(
            n=sample_demand_nodes, random_state=random_seed
        )
    else:
        sample_df = df_stations

    demand_coords = sample_df[["Latitude", "Longitude"]].to_numpy()
    demand_weights = sample_df["Power (kW)"].to_numpy()

    # 2. Candidate Grid Generation
    lat_min, lat_max = demand_coords[:, 0].min(), demand_coords[:, 0].max()
    lon_min, lon_max = demand_coords[:, 1].min(), demand_coords[:, 1].max()

    grid_lats = np.linspace(lat_min, lat_max, int(np.sqrt(grid_size)))
    grid_lons = np.linspace(lon_min, lon_max, int(np.sqrt(grid_size)))
    grid_mesh = np.meshgrid(grid_lats, grid_lons)
    candidate_coords = np.vstack(
        [grid_mesh[0].ravel(), grid_mesh[1].ravel()]
    ).T

    num_candidates = len(candidate_coords)

    # 3. Realistic Station Parameters (CapEx and Power capacity)
    # Tier 1: Fast DC Charging Hub (Cost: $50,000, Capacity: 300 kW)
    # Tier 2: Standard Dual Port (Cost: $25,000, Capacity: 120 kW)
    # Tier 3: Medium AC Destination (Cost: $10,000, Capacity: 50 kW)
    tier_types = np.random.choice([0, 1, 2], size=num_candidates, p=[0.25, 0.45, 0.30])
    cost_map = np.array([50000.0, 25000.0, 10000.0])
    capacity_map = np.array([300.0, 120.0, 50.0])

    candidate_costs = cost_map[tier_types]
    candidate_capacities = capacity_map[tier_types]

    # 4. Pairwise Distance Matrix (km)
    dist_matrix = haversine_distance_matrix(demand_coords, candidate_coords)

    return OptimizationDataset(
        demand_points=demand_coords,
        demand_weights=demand_weights,
        candidate_locations=candidate_coords,
        candidate_costs=candidate_costs,
        candidate_capacities=candidate_capacities,
        distance_matrix=dist_matrix,
    )
