# RF Optimization and Planning Tools 📡

Advanced RF optimization and planning toolkit for GSM, UMTS, LTE, and 5G networks with MIMO beamforming simulation and propagation modeling capabilities.

## 🎯 Overview

This toolkit provides comprehensive RF optimization and planning functionalities including:
- **MIMO Beamforming Simulation** - Advanced beamforming algorithms and analysis
- **RF Propagation Modeling** - Multiple propagation models for coverage prediction
- **Network Data Processing** - GSM/UMTS worst cell analysis and processing
- **5G/6G Technologies** - Massive MIMO, mmWave, and advanced beamforming

## 🚀 Features

### MIMO Beamforming (`mimo_beamforming_simulator.py`)
- **Digital Beamforming**: Delay-and-sum, MVDR, Zero-forcing, MMSE
- **Massive MIMO**: Support for 64+ antenna elements
- **Multi-user MIMO**: Zero-forcing and interference management
- **Channel Models**: Rayleigh, Rician fading with spatial correlation
- **Real-time Analysis**: SINR calculations and throughput optimization

### RF Propagation Simulation (`rf_propagation_simulator.py`)
- **Free Space Path Loss**: Basic propagation modeling
- **Hata-Okumura Model**: Urban/suburban/rural environments (150-1500 MHz)
- **COST-231 Model**: Extended frequency range (1500-2000 MHz)
- **Two-Ray Model**: Ground reflection effects
- **Coverage Prediction**: Power budget and link analysis

### Data Processing Tools (`rf_optimization_tool.py`)
- **GSM Worst Cells**: Automated PRS report processing
- **UMTS Worst Cells**: Multi-sheet Excel file merging
- **Hexadecimal Conversion**: LAC/TAC value conversion
- **File Management**: ZIP extraction and data consolidation

### Demo Applications (`mimo_demo.py`)
- **Interactive Beamforming**: Command-line interface
- **Beam Pattern Visualization**: Linear and polar plots
- **Multi-user Scenarios**: Performance analysis
- **Real-time Simulation**: Adaptive beamforming demonstration

## 📋 Requirements

```bash
# Python 3.8+
pip install numpy
pip install matplotlib
pip install scipy
pip install pandas
pip install openpyxl
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/i880/RF_OPTIM.git
cd RF_OPTIM
```

2. Install dependencies:
```bash
pip install -r requirements.txt  # Create this file with above packages
```

3. Make scripts executable:
```bash
chmod +x rf_optimization_tool.py
chmod +x mimo_demo.py
```

## 📖 Usage Examples

### MIMO Beamforming Demo
```bash
# Basic beamforming with beam pattern visualization
python3 mimo_demo.py --num-antennas 8 --target-angle 30 --beam-pattern

# Multi-user MIMO simulation
python3 mimo_demo.py --multi-user --user-angles -30 0 30 --num-antennas 16

# Custom frequency and array size
python3 mimo_demo.py --frequency 3500 --num-antennas 32 --target-angle 45
```

### RF Propagation Simulation
```bash
python3 -c "
from rf_propagation_simulator import *
import numpy as np

# Configure antenna system
config = AntennaConfig(
    frequency_mhz=1800,
    tx_power_dbm=43,
    tx_height_m=30,
    rx_height_m=1.5
)

env = Environment(environment_type='urban')
sim = RFPropagationSimulator(config, env)

# Calculate coverage at different distances
distances = np.linspace(0.1, 10, 100)
received_power = [sim.calculate_received_power(d, 'hata') for d in distances]
print(f'Coverage analysis complete for {len(distances)} points')
"
```

### Data Processing Tools
```bash
# Process GSM worst cells
./rf_optimization_tool.py gsm --input-dir ./gsm_data

# Process UMTS worst cells  
./rf_optimization_tool.py umts --input-dir ./umts_data

# Convert hex values
./rf_optimization_tool.py hex-convert --input-file TACLAC.txt
```

### Advanced MIMO Simulation
```python
from mimo_beamforming_simulator import *
import numpy as np

# Configure massive MIMO array
array_config = ArrayConfig(
    num_elements=64,
    element_spacing=0.5,
    array_type="planar",
    rows=8,
    cols=8
)

# Multi-user scenario
users = [
    UserConfig(position=(-30, 0), snr_db=15),
    UserConfig(position=(0, 0), snr_db=20),
    UserConfig(position=(30, 0), snr_db=12)
]

# Generate channel matrix
channel = MIMOChannel(64, 1, "rayleigh")
H = channel.generate_channel(num_users=3)

# Apply MMSE beamforming
weights = BeamformingAlgorithms.mmse_beamforming(H.squeeze(), noise_variance=0.1)
print(f"Beamforming weights shape: {weights.shape}")
```

## 📊 Key Algorithms

### Beamforming Methods
- **Delay-and-Sum**: `weights = exp(-j * 2π * d * sin(θ) / λ) / √N`
- **MVDR**: `w = R⁻¹h / (h^H R⁻¹ h)` where R is interference covariance
- **Zero-Forcing**: `W = H^† (Moore-Penrose pseudoinverse)`
- **MMSE**: `W = (H^H H + σ²I)⁻¹ H^H`

### Propagation Models
- **Free Space**: `FSPL = 32.45 + 20log₁₀(d) + 20log₁₀(f)`
- **Hata-Okumura**: `L₅₀ = 69.55 + 26.16log₁₀(f) - 13.82log₁₀(hₜₓ) - Cₕ + (44.9 - 6.55log₁₀(hₜₓ))log₁₀(d)`
- **COST-231**: Extended Hata model for 1500-2000 MHz

## 🏗️ Project Structure

```
RF_OPTIM/
├── README.md                          # This file
├── LICENSE                           # Apache 2.0 License
├── mimo_beamforming_simulator.py     # Advanced MIMO/beamforming algorithms
├── mimo_beamforming_fixed.py         # Fixed/optimized MIMO implementations
├── mimo_demo.py                      # Interactive beamforming demo
├── rf_propagation_simulator.py       # RF propagation models
├── rf_optimization_tool.py           # Data processing pipeline
├── rf_optimization.py                # Basic optimization functions
├── TOMORROW_TASKS.md                 # Development roadmap
└── __pycache__/                      # Python cache files
```

## 🔬 Technical Specifications

### Frequency Ranges
- **GSM**: 900/1800 MHz
- **UMTS**: 2100 MHz  
- **LTE**: 700-2600 MHz
- **5G NR**: 3.5 GHz, 28 GHz (mmWave)

### Antenna Configurations
- **Linear Arrays**: 2-128 elements
- **Planar Arrays**: Up to 64x64 elements
- **Element Spacing**: 0.5λ (configurable)

### Channel Models
- **Rayleigh Fading**: NLOS scenarios
- **Rician Fading**: LOS with K-factor
- **Spatial Correlation**: Configurable correlation matrices

## 📈 Performance Metrics

### MIMO Systems
- **SINR**: Signal-to-Interference-plus-Noise Ratio
- **Throughput**: Shannon capacity calculation
- **Spectral Efficiency**: bps/Hz per user
- **Array Gain**: Beamforming improvement

### Coverage Analysis
- **RSRP**: Reference Signal Received Power
- **RSRQ**: Reference Signal Received Quality  
- **Path Loss**: dB attenuation vs distance
- **Coverage Probability**: Statistical analysis

## 🚧 Current Development

See `TOMORROW_TASKS.md` for detailed development roadmap including:
- Machine learning integration
- 6G research concepts
- Software Defined Radio (SDR) support
- Real-time optimization algorithms

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**i880** - RF Optimization and Telecommunications Engineering

## 🙏 Acknowledgments

- Based on "Python For RF Optimization & Planning Engineers"
- Advanced 5G beamforming research
- Open source scientific computing community

## 📞 Support

For questions, issues, or feature requests:
- Open an issue on GitHub
- Contact: [Your contact information]

---

**⚡ Quick Start**: Run `python3 mimo_demo.py --multi-user --beam-pattern` for an interactive demo!

**🎯 Next Steps**: Check `TOMORROW_TASKS.md` for advanced features and development roadmap.

