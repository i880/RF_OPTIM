#!/usr/bin/env python3
"""
RF Propagation Model Simulator
Advanced RF optimization tool for path loss calculations and coverage prediction

This tool implements multiple propagation models:
- Free Space Path Loss
- Hata-Okumura Model
- COST-231 Model
- Terrain-based propagation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional
import json

@dataclass
class AntennaConfig:
    """Antenna configuration parameters"""
    frequency_mhz: float
    tx_power_dbm: float
    tx_height_m: float
    rx_height_m: float
    tx_gain_dbi: float = 0
    rx_gain_dbi: float = 0
    cable_loss_db: float = 0

@dataclass
class Environment:
    """Environment configuration"""
    environment_type: str = "urban"  # urban, suburban, rural
    terrain_height: Optional[np.ndarray] = None
    building_height: float = 20
    
class PropagationModels:
    """Collection of RF propagation models"""
    
    @staticmethod
    def free_space_path_loss(distance_km: float, frequency_mhz: float) -> float:
        """
        Calculate Free Space Path Loss
        
        Args:
            distance_km: Distance in kilometers
            frequency_mhz: Frequency in MHz
            
        Returns:
            Path loss in dB
        """
        if distance_km <= 0 or frequency_mhz <= 0:
            return float('inf')
        
        # FSPL = 20*log10(d) + 20*log10(f) + 32.45
        # where d is in km and f is in MHz
        fspl = 20 * np.log10(distance_km) + 20 * np.log10(frequency_mhz) + 32.45
        return fspl
    
    @staticmethod
    def hata_okumura_model(distance_km: float, 
                          frequency_mhz: float,
                          tx_height_m: float,
                          rx_height_m: float,
                          environment: str = "urban") -> float:
        """
        Calculate path loss using Hata-Okumura model
        
        Valid for:
        - Frequency: 150-1500 MHz
        - Distance: 1-20 km
        - Base station height: 30-200 m
        - Mobile height: 1-10 m
        """
        if not (150 <= frequency_mhz <= 1500):
            raise ValueError("Frequency must be between 150-1500 MHz for Hata model")
        
        if not (1 <= distance_km <= 20):
            raise ValueError("Distance must be between 1-20 km for Hata model")
        
        # Correction factor for mobile antenna height
        if environment.lower() == "urban":
            if frequency_mhz >= 400:
                C_h = 3.2 * (np.log10(11.75 * rx_height_m))**2 - 4.97
            else:
                C_h = 0.8 + (1.1 * np.log10(frequency_mhz) - 0.7) * rx_height_m - 1.56 * np.log10(frequency_mhz)
        else:
            C_h = 0.8 + (1.1 * np.log10(frequency_mhz) - 0.7) * rx_height_m - 1.56 * np.log10(frequency_mhz)
        
        # Basic path loss formula
        L50 = (69.55 + 26.16 * np.log10(frequency_mhz) - 13.82 * np.log10(tx_height_m) 
               - C_h + (44.9 - 6.55 * np.log10(tx_height_m)) * np.log10(distance_km))
        
        # Environment correction
        if environment.lower() == "suburban":
            L50 = L50 - 2 * (np.log10(frequency_mhz / 28))**2 - 5.4
        elif environment.lower() == "rural":
            L50 = L50 - 4.78 * (np.log10(frequency_mhz))**2 + 18.33 * np.log10(frequency_mhz) - 40.94
        
        return L50
    
    @staticmethod
    def cost231_model(distance_km: float,
                     frequency_mhz: float, 
                     tx_height_m: float,
                     rx_height_m: float,
                     environment: str = "urban") -> float:
        """
        Calculate path loss using COST-231 Hata model
        
        Valid for:
        - Frequency: 1500-2000 MHz
        - Distance: 1-20 km
        """
        if not (1500 <= frequency_mhz <= 2000):
            raise ValueError("Frequency must be between 1500-2000 MHz for COST-231 model")
        
        # Mobile antenna height correction factor
        C_h = 0.8 + (1.1 * np.log10(frequency_mhz) - 0.7) * rx_height_m - 1.56 * np.log10(frequency_mhz)
        
        # Environment factor
        C_m = 0 if environment.lower() in ["suburban", "rural"] else 3
        
        # COST-231 formula
        L50 = (46.3 + 33.9 * np.log10(frequency_mhz) - 13.82 * np.log10(tx_height_m)
               - C_h + (44.9 - 6.55 * np.log10(tx_height_m)) * np.log10(distance_km) + C_m)
        
        return L50
    
    @staticmethod
    def two_ray_model(distance_m: float,
                     frequency_mhz: float,
                     tx_height_m: float,
                     rx_height_m: float) -> float:
        """
        Calculate path loss using Two-Ray Ground Reflection model
        """
        wavelength = 3e8 / (frequency_mhz * 1e6)  # wavelength in meters
        
        # Critical distance
        d_critical = (4 * tx_height_m * rx_height_m) / wavelength
        
        if distance_m < d_critical:
            # Use free space model for short distances
            return PropagationModels.free_space_path_loss(distance_m/1000, frequency_mhz)
        else:
            # Two-ray model
            path_loss_linear = (distance_m**4) / ((tx_height_m * rx_height_m)**2)
            path_loss_db = 10 * np.log10(path_loss_linear)
            return path_loss_db

class RFPropagationSimulator:
    """Main RF Propagation Simulator class"""
    
    def __init__(self, antenna_config: AntennaConfig, environment: Environment):
        self.antenna_config = antenna_config
        self.environment = environment
        self.models = PropagationModels()
        
    def calculate_received_power(self, distance_km: float, model: str = "free_space") -> float:
        """
        Calculate received power based on propagation model
        
        Args:
            distance_km: Distance in kilometers
            model: Propagation model to use
            
        Returns:
            Received power in dBm
        """
        # Select propagation model
        if model == "free_space":
            path_loss = self.models.free_space_path_loss(distance_km, self.antenna_config.frequency_mhz)
        elif model == "hata":
            path_loss = self.models.hata_okumura_model(
                distance_km, 
                self.antenna_config.frequency_mhz,
                self.antenna_config.tx_height_m,
                self.antenna_config.rx_height_m,
                self.environment.environment_type
            )
        elif model == "cost231":
            path_loss = self.models.cost231_model(
                distance_km,
                self.antenna_config.frequency_mhz,
                self.antenna_config.tx_height_m,
                self.antenna_config.rx_height_m,
                self.environment.environment_type
            )
        elif model == "two_ray":
            path_loss = self.models.two_ray_model(
                distance_km * 1000,  # Convert to meters
                self.antenna_config.frequency_mhz,
                self.antenna_config.tx_height_m,
                self.antenna_config.rx_height_m
            )
        else:
            raise ValueError(f"Unknown model: {model}")
        
        # Calculate received power
        rx_power = (self.antenna_config.tx_power_dbm + 
                   self.antenna_config.tx_gain_dbi + 
                   self.antenna_config.rx_gain_dbi - 
                   self.antenna_config.cable_loss_db - 
                   path_loss)
        
        return rx_power
    
    def generate_coverage_map(self, 
                            center_lat: float = 0, 
                            center_lon: float = 0,
                            map_size_km: float = 10,
                            resolution: int = 100,
                            model: str = "free_space") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate coverage map around a transmitter
        
        Returns:
            X, Y coordinates and received power grid
        """
        # Create coordinate grids
        x = np.linspace(-map_size_km/2, map_size_km/2, resolution)
        y = np.linspace(-map_size_km/2, map_size_km/2, resolution)
        X, Y = np.meshgrid(x, y)
        
        # Calculate distances
        distances = np.sqrt(X**2 + Y**2)
        distances[distances == 0] = 0.001  # Avoid division by zero
        
        # Calculate received power for each point
        rx_power_grid = np.zeros_like(distances)
        for i in range(resolution):
            for j in range(resolution):
                if distances[i, j] > 0:
                    rx_power_grid[i, j] = self.calculate_received_power(distances[i, j], model)
        
        return X, Y, rx_power_grid
    
    def plot_coverage_map(self, X: np.ndarray, Y: np.ndarray, rx_power: np.ndarray, 
                         title: str = "RF Coverage Map"):
        """Plot the coverage map"""
        plt.figure(figsize=(12, 10))
        
        # Create contour plot
        levels = np.arange(-120, -30, 10)
        cs = plt.contourf(X, Y, rx_power, levels=levels, cmap='viridis', extend='both')
        plt.colorbar(cs, label='Received Power (dBm)')
        
        # Add contour lines
        cs_lines = plt.contour(X, Y, rx_power, levels=levels[::2], colors='white', linewidths=0.5)
        plt.clabel(cs_lines, inline=True, fontsize=8, fmt='%d dBm')
        
        # Mark transmitter location
        plt.plot(0, 0, 'r*', markersize=15, label='Transmitter')
        
        plt.xlabel('Distance (km)')
        plt.ylabel('Distance (km)')
        plt.title(f'{title}\n{self.antenna_config.frequency_mhz} MHz, {self.environment.environment_type} environment')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
    
    def compare_models(self, distances_km: np.ndarray) -> None:
        """Compare different propagation models"""
        models = ["free_space", "hata", "two_ray"]
        
        plt.figure(figsize=(12, 8))
        
        for model in models:
            rx_powers = []
            for distance in distances_km:
                try:
                    rx_power = self.calculate_received_power(distance, model)
                    rx_powers.append(rx_power)
                except ValueError:
                    rx_powers.append(None)
            
            # Filter out None values
            valid_distances = [d for d, p in zip(distances_km, rx_powers) if p is not None]
            valid_powers = [p for p in rx_powers if p is not None]
            
            plt.plot(valid_distances, valid_powers, 'o-', label=f'{model.upper()} Model', linewidth=2)
        
        plt.xlabel('Distance (km)')
        plt.ylabel('Received Power (dBm)')
        plt.title(f'Propagation Model Comparison\n{self.antenna_config.frequency_mhz} MHz, {self.environment.environment_type} environment')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def link_budget_analysis(self, distance_km: float, model: str = "free_space") -> dict:
        """Perform detailed link budget analysis"""
        rx_power = self.calculate_received_power(distance_km, model)
        
        # Calculate various parameters
        path_loss = getattr(self.models, f"{model}_path_loss" if hasattr(self.models, f"{model}_path_loss") else "free_space_path_loss")(
            distance_km, self.antenna_config.frequency_mhz
        )
        
        # Assume typical noise parameters
        noise_figure_db = 3  # Typical receiver noise figure
        bandwidth_mhz = 20   # Typical LTE bandwidth
        thermal_noise_dbm = -174 + 10 * np.log10(bandwidth_mhz * 1e6)  # Thermal noise power
        noise_power_dbm = thermal_noise_dbm + noise_figure_db
        
        snr_db = rx_power - noise_power_dbm
        
        return {
            "tx_power_dbm": self.antenna_config.tx_power_dbm,
            "tx_gain_dbi": self.antenna_config.tx_gain_dbi,
            "rx_gain_dbi": self.antenna_config.rx_gain_dbi,
            "cable_loss_db": self.antenna_config.cable_loss_db,
            "path_loss_db": path_loss,
            "rx_power_dbm": rx_power,
            "noise_power_dbm": noise_power_dbm,
            "snr_db": snr_db,
            "distance_km": distance_km,
            "frequency_mhz": self.antenna_config.frequency_mhz
        }

def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(description="RF Propagation Simulator")
    parser.add_argument('--frequency', type=float, default=1800, help='Frequency in MHz')
    parser.add_argument('--tx-power', type=float, default=43, help='TX power in dBm')
    parser.add_argument('--tx-height', type=float, default=30, help='TX antenna height in meters')
    parser.add_argument('--rx-height', type=float, default=1.5, help='RX antenna height in meters')
    parser.add_argument('--environment', choices=['urban', 'suburban', 'rural'], default='urban')
    parser.add_argument('--distance', type=float, default=5, help='Distance for link budget (km)')
    parser.add_argument('--model', choices=['free_space', 'hata', 'cost231', 'two_ray'], default='free_space')
    parser.add_argument('--coverage-map', action='store_true', help='Generate coverage map')
    parser.add_argument('--compare-models', action='store_true', help='Compare propagation models')
    parser.add_argument('--link-budget', action='store_true', help='Perform link budget analysis')
    
    args = parser.parse_args()
    
    # Create configuration objects
    antenna_config = AntennaConfig(
        frequency_mhz=args.frequency,
        tx_power_dbm=args.tx_power,
        tx_height_m=args.tx_height,
        rx_height_m=args.rx_height
    )
    
    environment = Environment(environment_type=args.environment)
    
    # Create simulator
    simulator = RFPropagationSimulator(antenna_config, environment)
    
    # Execute requested operations
    if args.coverage_map:
        print("Generating coverage map...")
        X, Y, rx_power = simulator.generate_coverage_map(model=args.model)
        simulator.plot_coverage_map(X, Y, rx_power, f"{args.model.upper()} Model Coverage Map")
    
    if args.compare_models:
        print("Comparing propagation models...")
        distances = np.logspace(-1, 1.3, 50)  # 0.1 to 20 km
        simulator.compare_models(distances)
    
    if args.link_budget:
        print(f"Performing link budget analysis at {args.distance} km...")
        budget = simulator.link_budget_analysis(args.distance, args.model)
        
        print("\n" + "="*50)
        print("LINK BUDGET ANALYSIS")
        print("="*50)
        for key, value in budget.items():
            if isinstance(value, float):
                print(f"{key.replace('_', ' ').title()}: {value:.2f}")
            else:
                print(f"{key.replace('_', ' ').title()}: {value}")
        print("="*50)
    
    if not any([args.coverage_map, args.compare_models, args.link_budget]):
        print("No analysis requested. Use --help for options.")

if __name__ == "__main__":
    main()
