#!/usr/bin/env python3
"""
MIMO Beamforming Simulator
Advanced 5G/6G beamforming algorithms and analysis

This tool implements:
- Digital Beamforming
- Analog Beamforming  
- Hybrid Beamforming
- Massive MIMO systems
- User tracking and beam steering
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import json

@dataclass
class ArrayConfig:
    """Antenna array configuration"""
    num_elements: int
    element_spacing: float = 0.5  # in wavelengths
    array_type: str = "linear"  # linear, planar, circular
    rows: int = 1  # for planar arrays
    cols: int = None  # for planar arrays

@dataclass
class UserConfig:
    """User/device configuration"""
    position: Tuple[float, float]  # (azimuth_deg, elevation_deg)
    velocity: Tuple[float, float] = (0, 0)  # velocity in degrees/second
    snr_db: float = 10
    data_rate_requirement: float = 100  # Mbps

class BeamformingAlgorithms:
    """Collection of beamforming algorithms"""
    
    @staticmethod
    def delay_and_sum(array_config: ArrayConfig, 
                     target_angle_deg: float) -> np.ndarray:
        """
        Simple delay-and-sum beamforming
        
        Args:
            array_config: Antenna array configuration
            target_angle_deg: Target angle in degrees
            
        Returns:
            Complex beamforming weights
        """
        target_angle_rad = np.deg2rad(target_angle_deg)
        
        if array_config.array_type == "linear":
            # Linear array positions
            element_positions = np.arange(array_config.num_elements) * array_config.element_spacing
            
            # Calculate phase shifts for steering
            phase_shifts = 2 * np.pi * element_positions * np.sin(target_angle_rad)
            
            # Beamforming weights
            weights = np.exp(-1j * phase_shifts) / np.sqrt(array_config.num_elements)
            
        elif array_config.array_type == "planar":
            rows = array_config.rows
            cols = array_config.cols or (array_config.num_elements // rows)
            
            # Planar array positions
            weights = np.zeros(rows * cols, dtype=complex)
            idx = 0
            for i in range(rows):
                for j in range(cols):
                    x_pos = j * array_config.element_spacing
                    y_pos = i * array_config.element_spacing
                    
                    phase_shift = 2 * np.pi * (x_pos * np.sin(target_angle_rad))
                    weights[idx] = np.exp(-1j * phase_shift)
                    idx += 1
            
            weights = weights / np.sqrt(len(weights))
        
        return weights
    
    @staticmethod
    def mvdr_beamforming(channel_matrix: np.ndarray,
                        noise_covariance: np.ndarray,
                        target_user_index: int = 0) -> np.ndarray:
        """
        Minimum Variance Distortionless Response (MVDR) beamforming
        
        Args:
            channel_matrix: Channel matrix (users x antennas)
            noise_covariance: Noise covariance matrix
            target_user_index: Index of target user
            
        Returns:
            MVDR beamforming weights
        """
        # Target user channel vector
        h_target = channel_matrix[target_user_index, :].reshape(-1, 1)
        
        # Calculate MVDR weights
        try:
            R_inv = np.linalg.inv(noise_covariance)
            numerator = R_inv @ h_target
            denominator = h_target.conj().T @ R_inv @ h_target
            weights = numerator / denominator
        except np.linalg.LinAlgError:
            # Fallback to delay-and-sum if matrix is singular
            weights = h_target.conj() / np.linalg.norm(h_target)
        
        return weights.flatten()
    
    @staticmethod
    def zf_beamforming(channel_matrix: np.ndarray) -> np.ndarray:
        """
        Zero Forcing (ZF) beamforming
        
        Args:
            channel_matrix: Channel matrix (users x antennas)
            
        Returns:
            ZF beamforming weights matrix
        """
        # Moore-Penrose pseudoinverse
        H_pinv = np.linalg.pinv(channel_matrix)
        return H_pinv.T
    
    @staticmethod
    def mmse_beamforming(channel_matrix: np.ndarray,
                        noise_variance: float = 0.1) -> np.ndarray:
        """
        Minimum Mean Square Error (MMSE) beamforming
        
        Args:
            channel_matrix: Channel matrix (users x antennas)
            noise_variance: Noise variance
            
        Returns:
            MMSE beamforming weights matrix
        """
        H = channel_matrix
        num_users, num_antennas = H.shape
        
        # MMSE solution
        HH_H = H.conj().T @ H
        regularization = noise_variance * np.eye(num_antennas)
        
        try:
            inv_term = np.linalg.inv(HH_H + regularization)
            weights = inv_term @ H.conj().T
        except np.linalg.LinAlgError:
            # Fallback to ZF if matrix is singular
            weights = BeamformingAlgorithms.zf_beamforming(channel_matrix)
        
        return weights.T

class MIMOChannel:
    """MIMO channel model"""
    
    def __init__(self, 
                 num_tx_antennas: int,
                 num_rx_antennas: int,
                 channel_type: str = "rayleigh"):
        self.num_tx = num_tx_antennas
        self.num_rx = num_rx_antennas
        self.channel_type = channel_type
    
    def generate_channel(self, 
                        num_users: int = 1,
                        correlation: float = 0.0) -> np.ndarray:
        """
        Generate MIMO channel matrix
        
        Args:
            num_users: Number of users
            correlation: Spatial correlation factor
            
        Returns:
            Channel matrix (users x tx_antennas x rx_antennas)
        """
        if self.channel_type == "rayleigh":
            # Rayleigh fading channel
            H_real = np.random.randn(num_users, self.num_rx, self.num_tx)
            H_imag = np.random.randn(num_users, self.num_rx, self.num_tx)
            H = (H_real + 1j * H_imag) / np.sqrt(2)
            
        elif self.channel_type == "rician":
            # Rician fading with K-factor = 3 dB
            K_factor = 2  # Linear scale
            
            # Line-of-sight component
            H_los = np.ones((num_users, self.num_rx, self.num_tx))
            
            # Scattered component
            H_scattered = (np.random.randn(num_users, self.num_rx, self.num_tx) + 
                          1j * np.random.randn(num_users, self.num_rx, self.num_tx)) / np.sqrt(2)
            
            # Combine LOS and scattered
            H = (np.sqrt(K_factor/(K_factor+1)) * H_los + 
                 np.sqrt(1/(K_factor+1)) * H_scattered)
        
        # Add spatial correlation if specified
        if correlation > 0:
            correlation_matrix = self._generate_correlation_matrix(self.num_tx, correlation)
            for user in range(num_users):
                H[user] = H[user] @ correlation_matrix
        
        return H
    
    def _generate_correlation_matrix(self, size: int, correlation: float) -> np.ndarray:
        """Generate exponential correlation matrix"""
        R = np.zeros((size, size), dtype=complex)
        for i in range(size):
            for j in range(size):
                R[i, j] = correlation ** abs(i - j)
        return R

class MIMOBeamformingSimulator:
    """Main MIMO beamforming simulator"""
    
    def __init__(self, 
                 array_config: ArrayConfig,
                 channel_model: MIMOChannel):
        self.array_config = array_config
        self.channel_model = channel_model
        self.algorithms = BeamformingAlgorithms()
        
    def calculate_array_factor(self, 
                              weights: np.ndarray,
                              angles_deg: np.ndarray) -> np.ndarray:
        """
        Calculate array factor for given weights and angles
        
        Args:
            weights: Beamforming weights
            angles_deg: Array of angles in degrees
            
        Returns:
            Array factor magnitude in dB
        """
        angles_rad = np.deg2rad(angles_deg)
        
        if self.array_config.array_type == "linear":
            # Linear array factor
            element_positions = np.arange(self.array_config.num_elements) * self.array_config.element_spacing
            
            array_factor = np.zeros(len(angles_deg), dtype=complex)
            for i, angle in enumerate(angles_rad):\n                steering_vector = np.exp(1j * 2 * np.pi * element_positions * np.sin(angle))\n                array_factor[i] = np.sum(weights * steering_vector)\n        \n        # Convert to dB\n        array_factor_db = 20 * np.log10(np.abs(array_factor) + 1e-10)\n        \n        return array_factor_db\n    \n    def plot_beam_pattern(self, \n                         weights: np.ndarray,\n                         title: str = \"Beam Pattern\",\n                         target_angle: Optional[float] = None):\n        \"\"\"Plot the beam pattern\"\"\"\n        angles = np.linspace(-90, 90, 361)\n        array_factor = self.calculate_array_factor(weights, angles)\n        \n        plt.figure(figsize=(10, 6))\n        plt.plot(angles, array_factor, 'b-', linewidth=2)\n        plt.xlabel('Angle (degrees)')\n        plt.ylabel('Array Factor (dB)')\n        plt.title(title)\n        plt.grid(True, alpha=0.3)\n        \n        # Mark target angle if provided\n        if target_angle is not None:\n            plt.axvline(x=target_angle, color='r', linestyle='--', \n                       label=f'Target: {target_angle}°')\n            plt.legend()\n        \n        plt.ylim([-40, 5])\n        plt.tight_layout()\n        plt.show()\n    \n    def simulate_multi_user_mimo(self, \n                                 user_configs: List[UserConfig],\n                                 algorithm: str = \"zf\") -> Dict:\n        \"\"\"Simulate multi-user MIMO system\"\"\"\n        num_users = len(user_configs)\n        num_antennas = self.array_config.num_elements\n        \n        # Generate channel matrix\n        H = self.channel_model.generate_channel(num_users)\n        H_2d = H.reshape(num_users, num_antennas)  # Flatten to 2D\n        \n        # Calculate beamforming weights based on algorithm\n        if algorithm == \"zf\":\n            W = self.algorithms.zf_beamforming(H_2d)\n        elif algorithm == \"mmse\":\n            W = self.algorithms.mmse_beamforming(H_2d)\n        elif algorithm == \"mvdr\":\n            # For MVDR, we need noise covariance\n            noise_covariance = 0.1 * np.eye(num_antennas)\n            W = np.zeros((num_antennas, num_users), dtype=complex)\n            for user in range(num_users):\n                W[:, user] = self.algorithms.mvdr_beamforming(H_2d, noise_covariance, user)\n        else:\n            raise ValueError(f\"Unknown algorithm: {algorithm}\")\n        \n        # Calculate performance metrics\n        results = {\n            \"channel_matrix\": H_2d,\n            \"beamforming_weights\": W,\n            \"sinr_per_user\": [],\n            \"throughput_per_user\": [],\n            \"total_throughput\": 0\n        }\n        \n        # Calculate SINR for each user\n        for user in range(num_users):\n            h_user = H_2d[user, :].reshape(1, -1)\n            w_user = W[:, user].reshape(-1, 1)\n            \n            # Desired signal power\n            signal_power = np.abs(h_user @ w_user)**2\n            \n            # Interference power from other users\n            interference_power = 0\n            for other_user in range(num_users):\n                if other_user != user:\n                    w_other = W[:, other_user].reshape(-1, 1)\n                    interference_power += np.abs(h_user @ w_other)**2\n            \n            # Noise power (normalized)\n            noise_power = 0.1\n            \n            # SINR calculation\n            sinr_linear = signal_power / (interference_power + noise_power)\n            sinr_db = 10 * np.log10(sinr_linear.real)\n            \n            # Shannon capacity (throughput)\n            throughput = np.log2(1 + sinr_linear.real) * 20  # 20 MHz bandwidth\n            \n            results[\"sinr_per_user\"].append(sinr_db.item())\n            results[\"throughput_per_user\"].append(throughput.item())\n        \n        results[\"total_throughput\"] = sum(results[\"throughput_per_user\"])\n        \n        return results\n    \n    def adaptive_beamforming_simulation(self, \n                                       user_configs: List[UserConfig],\n                                       time_steps: int = 100,\n                                       update_interval: int = 10) -> Dict:\n        \"\"\"Simulate adaptive beamforming with moving users\"\"\"\n        num_users = len(user_configs)\n        num_antennas = self.array_config.num_elements\n        \n        # Initialize tracking arrays\n        sinr_history = [[] for _ in range(num_users)]\n        throughput_history = []\n        beam_angles = [[] for _ in range(num_users)]\n        \n        for t in range(time_steps):\n            # Update user positions based on velocity\n            current_positions = []\n            for i, user in enumerate(user_configs):\n                new_azimuth = user.position[0] + user.velocity[0] * t * 0.1\n                new_elevation = user.position[1] + user.velocity[1] * t * 0.1\n                current_positions.append((new_azimuth, new_elevation))\n                beam_angles[i].append(new_azimuth)\n            \n            # Update beamforming weights periodically\n            if t % update_interval == 0:\n                # Generate new channel based on current positions\n                H = self.channel_model.generate_channel(num_users)\n                H_2d = H.reshape(num_users, num_antennas)\n                \n                # Recalculate beamforming weights\n                W = self.algorithms.zf_beamforming(H_2d)\n            \n            # Calculate current performance\n            results = self.simulate_multi_user_mimo(user_configs, \"zf\")\n            \n            for i in range(num_users):\n                sinr_history[i].append(results[\"sinr_per_user\"][i])\n            \n            throughput_history.append(results[\"total_throughput\"])\n        \n        return {\n            \"sinr_history\": sinr_history,\n            \"throughput_history\": throughput_history,\n            \"beam_angles\": beam_angles,\n            \"time_steps\": time_steps\n        }\n    \n    def plot_adaptive_results(self, results: Dict):\n        \"\"\"Plot adaptive beamforming results\"\"\"\n        time_axis = np.arange(results[\"time_steps\"])\n        \n        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))\n        \n        # SINR over time\n        for i, sinr_hist in enumerate(results[\"sinr_history\"]):\n            ax1.plot(time_axis, sinr_hist, label=f'User {i+1}')\n        ax1.set_xlabel('Time Steps')\n        ax1.set_ylabel('SINR (dB)')\n        ax1.set_title('SINR vs Time')\n        ax1.legend()\n        ax1.grid(True, alpha=0.3)\n        \n        # Total throughput over time\n        ax2.plot(time_axis, results[\"throughput_history\"], 'b-', linewidth=2)\n        ax2.set_xlabel('Time Steps')\n        ax2.set_ylabel('Total Throughput (Mbps)')\n        ax2.set_title('System Throughput vs Time')\n        ax2.grid(True, alpha=0.3)\n        \n        # Beam steering angles\n        for i, angles in enumerate(results[\"beam_angles\"]):\n            ax3.plot(time_axis, angles, label=f'User {i+1}')\n        ax3.set_xlabel('Time Steps')\n        ax3.set_ylabel('Beam Angle (degrees)')\n        ax3.set_title('Beam Steering Angles')\n        ax3.legend()\n        ax3.grid(True, alpha=0.3)\n        \n        # Performance statistics\n        avg_sinr = [np.mean(sinr_hist) for sinr_hist in results[\"sinr_history\"]]\n        ax4.bar(range(len(avg_sinr)), avg_sinr)\n        ax4.set_xlabel('User Index')\n        ax4.set_ylabel('Average SINR (dB)')\n        ax4.set_title('Average SINR per User')\n        ax4.grid(True, alpha=0.3)\n        \n        plt.tight_layout()\n        plt.show()\n\ndef main():\n    \"\"\"Main function with CLI interface\"\"\"\n    parser = argparse.ArgumentParser(description=\"MIMO Beamforming Simulator\")\n    parser.add_argument('--num-antennas', type=int, default=8, help='Number of antennas')\n    parser.add_argument('--num-users', type=int, default=2, help='Number of users')\n    parser.add_argument('--array-type', choices=['linear', 'planar'], default='linear')\n    parser.add_argument('--algorithm', choices=['zf', 'mmse', 'mvdr'], default='zf')\n    parser.add_argument('--channel-type', choices=['rayleigh', 'rician'], default='rayleigh')\n    parser.add_argument('--beam-pattern', action='store_true', help='Show beam pattern')\n    parser.add_argument('--multi-user', action='store_true', help='Simulate multi-user MIMO')\n    parser.add_argument('--adaptive', action='store_true', help='Simulate adaptive beamforming')\n    parser.add_argument('--target-angle', type=float, default=30, help='Target beam angle (degrees)')\n    \n    args = parser.parse_args()\n    \n    # Create configuration objects\n    array_config = ArrayConfig(\n        num_elements=args.num_antennas,\n        array_type=args.array_type\n    )\n    \n    if args.array_type == \"planar\":\n        array_config.rows = int(np.sqrt(args.num_antennas))\n        array_config.cols = args.num_antennas // array_config.rows\n    \n    channel_model = MIMOChannel(\n        num_tx_antennas=args.num_antennas,\n        num_rx_antennas=1,  # Single antenna per user\n        channel_type=args.channel_type\n    )\n    \n    # Create simulator\n    simulator = MIMOBeamformingSimulator(array_config, channel_model)\n    \n    # Execute requested simulations\n    if args.beam_pattern:\n        print(f\"Generating beam pattern for {args.target_angle}° target...\")\n        weights = simulator.algorithms.delay_and_sum(array_config, args.target_angle)\n        simulator.plot_beam_pattern(weights, \n                                   f\"Delay-and-Sum Beam Pattern ({args.num_antennas} elements)\",\n                                   args.target_angle)\n    \n    if args.multi_user:\n        print(f\"Simulating multi-user MIMO with {args.num_users} users...\")\n        \n        # Create user configurations\n        user_configs = []\n        angles = np.linspace(-45, 45, args.num_users)\n        for i, angle in enumerate(angles):\n            user_config = UserConfig(\n                position=(angle, 0),\n                snr_db=10 + i * 2\n            )\n            user_configs.append(user_config)\n        \n        results = simulator.simulate_multi_user_mimo(user_configs, args.algorithm)\n        \n        print(\"\\n\" + \"=\"*60)\n        print(\"MULTI-USER MIMO RESULTS\")\n        print(\"=\"*60)\n        print(f\"Algorithm: {args.algorithm.upper()}\")\n        print(f\"Number of antennas: {args.num_antennas}\")\n        print(f\"Number of users: {args.num_users}\")\n        print(\"-\"*60)\n        for i, (sinr, throughput) in enumerate(zip(results[\"sinr_per_user\"], results[\"throughput_per_user\"])):\n            print(f\"User {i+1}: SINR = {sinr:.2f} dB, Throughput = {throughput:.2f} Mbps\")\n        print(\"-\"*60)\n        print(f\"Total Throughput: {results['total_throughput']:.2f} Mbps\")\n        print(\"=\"*60)\n    \n    if args.adaptive:\n        print(\"Simulating adaptive beamforming...\")\n        \n        # Create moving users\n        user_configs = [\n            UserConfig(position=(-30, 0), velocity=(1, 0)),  # Moving right\n            UserConfig(position=(30, 0), velocity=(-0.5, 0))  # Moving left\n        ]\n        \n        results = simulator.adaptive_beamforming_simulation(user_configs, time_steps=100)\n        simulator.plot_adaptive_results(results)\n    \n    if not any([args.beam_pattern, args.multi_user, args.adaptive]):\n        print(\"No simulation requested. Use --help for options.\")\n\nif __name__ == \"__main__\":\n    main()
