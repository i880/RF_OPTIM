#!/usr/bin/env python3
"""
Simple MIMO Beamforming Demo
Demonstrates basic beamforming concepts
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse

def delay_and_sum_beamforming(num_antennas, target_angle_deg, frequency_mhz=2400):
    """
    Implement delay-and-sum beamforming
    
    Args:
        num_antennas: Number of antenna elements
        target_angle_deg: Target steering angle in degrees
        frequency_mhz: Operating frequency in MHz
    
    Returns:
        weights: Complex beamforming weights
    """
    wavelength = 3e8 / (frequency_mhz * 1e6)  # wavelength in meters
    element_spacing = wavelength / 2  # half-wavelength spacing
    
    # Convert angle to radians
    target_angle_rad = np.deg2rad(target_angle_deg)
    
    # Calculate element positions
    element_positions = np.arange(num_antennas) * element_spacing
    
    # Calculate phase shifts for steering to target angle
    phase_shifts = 2 * np.pi * element_positions * np.sin(target_angle_rad) / wavelength
    
    # Beamforming weights (complex exponentials)
    weights = np.exp(-1j * phase_shifts) / np.sqrt(num_antennas)
    
    return weights

def calculate_array_factor(weights, angles_deg, frequency_mhz=2400):
    """
    Calculate array factor for given weights and angles
    
    Args:
        weights: Complex beamforming weights
        angles_deg: Array of angles in degrees
        frequency_mhz: Operating frequency in MHz
    
    Returns:
        Array factor in dB
    """
    wavelength = 3e8 / (frequency_mhz * 1e6)
    element_spacing = wavelength / 2
    num_antennas = len(weights)
    
    angles_rad = np.deg2rad(angles_deg)
    element_positions = np.arange(num_antennas) * element_spacing
    
    array_factor = np.zeros(len(angles_deg), dtype=complex)
    
    for i, angle in enumerate(angles_rad):
        # Steering vector for this angle
        steering_vector = np.exp(1j * 2 * np.pi * element_positions * np.sin(angle) / wavelength)
        # Array factor is the dot product of weights and steering vector
        array_factor[i] = np.sum(weights * steering_vector)
    
    # Convert to dB
    array_factor_db = 20 * np.log10(np.abs(array_factor) + 1e-10)
    
    return array_factor_db

def plot_beam_pattern(array_factor_db, angles_deg, target_angle, num_antennas):
    """Plot the beam pattern"""
    plt.figure(figsize=(12, 8))
    
    # Linear plot
    plt.subplot(2, 1, 1)
    plt.plot(angles_deg, array_factor_db, 'b-', linewidth=2)
    plt.axvline(x=target_angle, color='r', linestyle='--', linewidth=2, label=f'Target: {target_angle}°')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Array Factor (dB)')
    plt.title(f'Beam Pattern - {num_antennas} Element Linear Array')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim([-40, 5])
    
    # Polar plot
    plt.subplot(2, 1, 2, projection='polar')
    angles_rad = np.deg2rad(angles_deg)
    # Normalize for polar plot (make minimum 0 dB)
    normalized_pattern = array_factor_db - np.min(array_factor_db)
    plt.plot(angles_rad, normalized_pattern, 'b-', linewidth=2)
    plt.fill(angles_rad, normalized_pattern, alpha=0.2)
    plt.title('Polar Beam Pattern')
    
    plt.tight_layout()
    plt.show()

def simulate_multi_user_scenario(num_antennas=8, user_angles=[-30, 0, 30]):
    """
    Simulate multi-user MIMO with zero-forcing beamforming
    
    Args:
        num_antennas: Number of transmit antennas
        user_angles: List of user angles in degrees
    """
    num_users = len(user_angles)
    frequency_mhz = 2400
    wavelength = 3e8 / (frequency_mhz * 1e6)
    element_spacing = wavelength / 2
    
    print(f"\n{'='*60}")
    print("MULTI-USER MIMO SIMULATION")
    print(f"{'='*60}")
    print(f"Number of antennas: {num_antennas}")
    print(f"Number of users: {num_users}")
    print(f"User angles: {user_angles}")
    print(f"Frequency: {frequency_mhz} MHz")
    
    # Generate channel matrix based on user positions
    element_positions = np.arange(num_antennas) * element_spacing
    H = np.zeros((num_users, num_antennas), dtype=complex)
    
    for i, angle in enumerate(user_angles):
        angle_rad = np.deg2rad(angle)
        # Channel vector for user i (steering vector)
        H[i, :] = np.exp(1j * 2 * np.pi * element_positions * np.sin(angle_rad) / wavelength)
        # Add some random fading
        H[i, :] *= (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
    
    print(f"\nChannel matrix shape: {H.shape}")
    print(f"Channel condition number: {np.linalg.cond(H):.2f}")
    
    # Zero-forcing beamforming
    try:
        W = np.linalg.pinv(H)  # Pseudo-inverse for beamforming weights
        print(f"Beamforming weights shape: {W.shape}")
        
        # Calculate SINR for each user
        print(f"\n{'-'*60}")
        print("USER PERFORMANCE")
        print(f"{'-'*60}")
        
        total_throughput = 0
        for user in range(num_users):
            h_user = H[user, :].reshape(1, -1)
            w_user = W[:, user].reshape(-1, 1)
            
            # Desired signal power
            signal_power = np.abs(h_user @ w_user)**2
            
            # Interference from other users
            interference_power = 0
            for other_user in range(num_users):
                if other_user != user:
                    w_other = W[:, other_user].reshape(-1, 1)
                    interference_power += np.abs(h_user @ w_other)**2
            
            # Noise power (assume -100 dBm)
            noise_power = 0.1
            
            # SINR calculation
            sinr_linear = signal_power / (interference_power + noise_power)
            sinr_db = 10 * np.log10(sinr_linear.real)
            
            # Shannon capacity (20 MHz bandwidth)
            throughput = np.log2(1 + sinr_linear.real) * 20
            total_throughput += throughput.item()
            
            print(f"User {user+1} (@ {user_angles[user]:+3.0f}°): "
                  f"SINR = {sinr_db.item():6.2f} dB, "
                  f"Throughput = {throughput.item():6.2f} Mbps")
        
        print(f"{'-'*60}")
        print(f"Total System Throughput: {total_throughput:.2f} Mbps")
        print(f"{'='*60}")
        
        return W, H
        
    except np.linalg.LinAlgError:
        print("Error: Cannot compute beamforming weights (singular matrix)")
        return None, None

def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(description="MIMO Beamforming Demo")
    parser.add_argument('--num-antennas', type=int, default=8, help='Number of antennas')
    parser.add_argument('--target-angle', type=float, default=30, help='Target beam angle (degrees)')
    parser.add_argument('--frequency', type=float, default=2400, help='Frequency in MHz')
    parser.add_argument('--beam-pattern', action='store_true', help='Show beam pattern')
    parser.add_argument('--multi-user', action='store_true', help='Simulate multi-user MIMO')
    parser.add_argument('--user-angles', nargs='+', type=float, default=[-30, 0, 30], 
                       help='User angles for multi-user simulation')
    
    args = parser.parse_args()
    
    print(f"MIMO Beamforming Demo")
    print(f"Frequency: {args.frequency} MHz")
    print(f"Number of antennas: {args.num_antennas}")
    
    if args.beam_pattern:
        print(f"\nGenerating beam pattern for {args.target_angle}° target...")
        
        # Calculate beamforming weights
        weights = delay_and_sum_beamforming(
            args.num_antennas, 
            args.target_angle, 
            args.frequency
        )
        
        # Calculate array factor
        angles = np.linspace(-90, 90, 361)
        array_factor = calculate_array_factor(weights, angles, args.frequency)
        
        # Plot results
        plot_beam_pattern(array_factor, angles, args.target_angle, args.num_antennas)
        
        # Print beam characteristics
        max_gain_idx = np.argmax(array_factor)
        max_gain = array_factor[max_gain_idx]
        max_gain_angle = angles[max_gain_idx]
        
        # Find 3dB beamwidth
        half_power_level = max_gain - 3
        half_power_indices = np.where(array_factor >= half_power_level)[0]
        if len(half_power_indices) > 0:
            beamwidth = angles[half_power_indices[-1]] - angles[half_power_indices[0]]
        else:
            beamwidth = 0
        
        print(f"\nBeam Characteristics:")
        print(f"  Max gain: {max_gain:.2f} dB at {max_gain_angle:.1f}°")
        print(f"  3dB beamwidth: {beamwidth:.1f}°")
        print(f"  Target angle: {args.target_angle}°")
        print(f"  Steering error: {abs(max_gain_angle - args.target_angle):.1f}°")
    
    if args.multi_user:
        print(f"\nRunning multi-user MIMO simulation...")
        simulate_multi_user_scenario(args.num_antennas, args.user_angles)
    
    if not any([args.beam_pattern, args.multi_user]):
        print("\nNo simulation requested. Use --help for options.")
        print("Example usage:")
        print("  python3 mimo_demo.py --beam-pattern --target-angle 45")
        print("  python3 mimo_demo.py --multi-user --user-angles -45 0 45")

if __name__ == "__main__":
    main()
