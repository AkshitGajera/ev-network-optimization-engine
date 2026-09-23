"""
Electric Vehicle Routing Problem (EVRP) with Charging Stops & Battery SoC Dynamics.
Operational-tier optimization: Plans fleet delivery tours visiting demand points,
monitoring battery State of Charge (SoC), and dispatching vehicles to optimized
charging stations before depleting battery.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from src.data_processing import haversine_distance_matrix


@dataclass
class RouteStop:
    node_id: int
    node_type: str  # 'depot', 'customer', 'charging_station'
    lat: float
    lon: float
    arrival_soc_pct: float
    departure_soc_pct: float
    distance_from_prev_km: float


@dataclass
class FleetTour:
    vehicle_id: int
    stops: List[RouteStop]
    total_distance_km: float
    charging_stops_count: int
    is_feasible: bool


class ElectricVehicleRouter:
    """Solves Electric Vehicle Routing Problem with Charging (EVRP).

    Features:
    - Battery capacity (kWh) and consumption rate (kWh/km)
    - Safety threshold (e.g. must not drop below 15% SoC)
    - Recharging stops at the nearest optimized charging station when required.
    """

    def __init__(
        self,
        depot_coords: Tuple[float, float],
        customer_coords: np.ndarray,
        station_coords: np.ndarray,
        battery_capacity_kwh: float = 65.0,
        consumption_kwh_per_km: float = 0.22,
        min_safe_soc_pct: float = 15.0,
        target_charge_soc_pct: float = 85.0,
    ):
        self.depot = np.array(depot_coords)
        self.customers = customer_coords
        self.stations = station_coords
        self.battery_cap = battery_capacity_kwh
        self.consumption = consumption_kwh_per_km
        self.min_safe_soc = min_safe_soc_pct
        self.target_charge_soc = target_charge_soc_pct

        # Max safe operational range without recharging
        usable_soc_fraction = (100.0 - self.min_safe_soc) / 100.0
        self.max_safe_range_km = (self.battery_cap * usable_soc_fraction) / self.consumption

    def _find_nearest_station(self, current_coord: np.ndarray) -> Tuple[int, float]:
        """Finds nearest charging station to current coordinate."""
        if len(self.stations) == 0:
            return -1, float("inf")
        dists = haversine_distance_matrix(current_coord[np.newaxis, :], self.stations)[0]
        nearest_idx = int(np.argmin(dists))
        return nearest_idx, dists[nearest_idx]

    def plan_fleet_routes(
        self,
        num_vehicles: int = 3,
        cluster_customers: bool = True,
    ) -> List[FleetTour]:
        """Plans multi-vehicle delivery routes from the depot visiting all customers

        with automated en-route charging detour insertion when battery level reaches safety threshold.
        """
        num_cust = len(self.customers)
        if num_cust == 0:
            return []

        # Partition customers into vehicle clusters
        if cluster_customers and num_cust >= num_vehicles:
            # Sort customers by angle relative to depot (sweep algorithm)
            angles = np.arctan2(
                self.customers[:, 0] - self.depot[0],
                self.customers[:, 1] - self.depot[1],
            )
            sorted_indices = np.argsort(angles)
            customer_batches = np.array_split(sorted_indices, num_vehicles)
        else:
            customer_batches = np.array_split(np.arange(num_cust), num_vehicles)

        tours: List[FleetTour] = []

        for v_id, batch in enumerate(customer_batches):
            if len(batch) == 0:
                continue

            stops: List[RouteStop] = []
            curr_coord = self.depot
            curr_soc = 100.0  # Start fully charged
            total_dist = 0.0
            recharge_count = 0
            feasible = True

            # Initial depot stop
            stops.append(
                RouteStop(
                    node_id=0,
                    node_type="depot",
                    lat=curr_coord[0],
                    lon=curr_coord[1],
                    arrival_soc_pct=100.0,
                    departure_soc_pct=100.0,
                    distance_from_prev_km=0.0,
                )
            )

            # Visit customers in the cluster (Nearest Neighbor Order)
            unvisited = list(batch)
            while unvisited:
                # Find nearest unvisited customer
                dists_to_unvisited = [
                    haversine_distance_matrix(
                        curr_coord[np.newaxis, :], self.customers[idx : idx + 1]
                    )[0, 0]
                    for idx in unvisited
                ]
                best_local_idx = int(np.argmin(dists_to_unvisited))
                next_cust_idx = unvisited.pop(best_local_idx)
                next_coord = self.customers[next_cust_idx]
                leg_dist = dists_to_unvisited[best_local_idx]

                # Calculate projected SoC after visiting this customer
                energy_needed_kwh = leg_dist * self.consumption
                soc_drop = (energy_needed_kwh / self.battery_cap) * 100.0
                projected_soc = curr_soc - soc_drop

                # If battery would drop below safety margin, insert charging stop detour
                if projected_soc < self.min_safe_soc and len(self.stations) > 0:
                    st_idx, st_dist = self._find_nearest_station(curr_coord)
                    st_coord = self.stations[st_idx]

                    # Travel to charging station
                    st_energy = st_dist * self.consumption
                    st_soc_drop = (st_energy / self.battery_cap) * 100.0
                    arr_st_soc = curr_soc - st_soc_drop

                    if arr_st_soc < 0:
                        feasible = False  # Battery ran out before reaching station

                    total_dist += st_dist
                    recharge_count += 1

                    stops.append(
                        RouteStop(
                            node_id=st_idx,
                            node_type="charging_station",
                            lat=st_coord[0],
                            lon=st_coord[1],
                            arrival_soc_pct=max(0.0, arr_st_soc),
                            departure_soc_pct=self.target_charge_soc,
                            distance_from_prev_km=st_dist,
                        )
                    )

                    # Reset SoC after fast charge
                    curr_coord = st_coord
                    curr_soc = self.target_charge_soc

                    # Now recompute distance from station to customer
                    leg_dist = haversine_distance_matrix(
                        curr_coord[np.newaxis, :], next_coord[np.newaxis, :]
                    )[0, 0]
                    energy_needed_kwh = leg_dist * self.consumption
                    soc_drop = (energy_needed_kwh / self.battery_cap) * 100.0
                    projected_soc = curr_soc - soc_drop

                # Move to customer
                total_dist += leg_dist
                curr_soc = projected_soc
                curr_coord = next_coord

                stops.append(
                    RouteStop(
                        node_id=next_cust_idx,
                        node_type="customer",
                        lat=curr_coord[0],
                        lon=curr_coord[1],
                        arrival_soc_pct=max(0.0, curr_soc),
                        departure_soc_pct=max(0.0, curr_soc),
                        distance_from_prev_km=leg_dist,
                    )
                )

            # Return to depot
            return_dist = haversine_distance_matrix(
                curr_coord[np.newaxis, :], self.depot[np.newaxis, :]
            )[0, 0]
            return_soc_drop = (return_dist * self.consumption / self.battery_cap) * 100.0
            final_soc = curr_soc - return_soc_drop

            # Check if charging detour needed to reach home depot
            if final_soc < self.min_safe_soc and len(self.stations) > 0:
                st_idx, st_dist = self._find_nearest_station(curr_coord)
                st_coord = self.stations[st_idx]
                total_dist += st_dist
                recharge_count += 1
                stops.append(
                    RouteStop(
                        node_id=st_idx,
                        node_type="charging_station",
                        lat=st_coord[0],
                        lon=st_coord[1],
                        arrival_soc_pct=max(0.0, curr_soc - (st_dist * self.consumption / self.battery_cap) * 100.0),
                        departure_soc_pct=self.target_charge_soc,
                        distance_from_prev_km=st_dist,
                    )
                )
                curr_coord = st_coord
                curr_soc = self.target_charge_soc
                return_dist = haversine_distance_matrix(
                    curr_coord[np.newaxis, :], self.depot[np.newaxis, :]
                )[0, 0]
                final_soc = curr_soc - (return_dist * self.consumption / self.battery_cap) * 100.0

            total_dist += return_dist
            stops.append(
                RouteStop(
                    node_id=0,
                    node_type="depot",
                    lat=self.depot[0],
                    lon=self.depot[1],
                    arrival_soc_pct=max(0.0, final_soc),
                    departure_soc_pct=100.0,
                    distance_from_prev_km=return_dist,
                )
            )

            tours.append(
                FleetTour(
                    vehicle_id=v_id + 1,
                    stops=stops,
                    total_distance_km=round(total_dist, 2),
                    charging_stops_count=recharge_count,
                    is_feasible=feasible and (final_soc >= 0),
                )
            )

        return tours
