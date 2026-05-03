# BB84 QKD Simulator - Web Interface Guide

## Quick Start

### 1. Install Streamlit and Dependencies

```bash
pip install streamlit plotly pandas
```

Or if you're using a virtual environment:
```bash
# Activate your environment first
python -m pip install streamlit plotly pandas
```

### 2. Run the Interface

```bash
streamlit run app.py
```

This will open a web browser at `http://localhost:8501`

### 3. What You Can Do

The interface has 5 main pages:

#### 🏠 **Home**
- Introduction to BB84 protocol
- Quick start guide
- Protocol flow diagram

#### 🎯 **Single Simulation**
- Run one BB84 protocol instance
- Configure:
  - Number of qubits (100-5000)
  - Eve's attack (on/off and intercept probability)
  - Channel noise (depolarizing, phase damping, amplitude damping)
  - Noise probability
- View detailed results:
  - QBER (Quantum Bit Error Rate)
  - Key rates
  - Security status
  - Confidence intervals

#### 📊 **Scenario Comparison**
- Compare multiple pre-defined scenarios:
  1. Ideal (no noise, no Eve)
  2. Eve - Full Intercept (100%)
  3. Eve - Half Intercept (50%)
  4. Noise (10% depolarizing, no Eve)
  5. Eve + Noise (combined)
- View results in table and multi-panel charts

#### 📈 **Eve Analysis**
- Sweep Eve's intercept probability from 0% to 100%
- Observe how QBER changes with eavesdropping intensity
- Identify detection thresholds

#### ℹ️ **About**
- Protocol explanation
- Security thresholds
- Technical details

---

## 🔐 Security Thresholds

| Status | QBER Range | Meaning |
|--------|-----------|---------|
| 🟢 SECURE | 0% – 5% | No eavesdropping detected |
| 🟡 WARNING | 5% – 11% | Possible attack or high noise |
| 🔴 ABORT | 11%+ | Likely eavesdropping - reject key |

---

## 📊 Output Metrics

- **Qubits Transmitted:** Total qubits Alice sends
- **Sifted Key:** Bits where Alice & Bob used same basis
- **Key Rate:** Sifted key length / transmitted qubits
- **QBER:** Quantum bit error rate (target: <5%)
- **Secret Key Length:** Usable key after QBER check
- **95% CI:** Confidence interval for QBER estimate

---

## ⚙️ Advanced Configuration

Edit `bb84config.py` to change defaults:

```python
@dataclass
class SimulationConfig:
    n_qubits            = 1000      # qubits to transmit
    eve_present         = False     # Eve active?
    eve_intercept_prob  = 1.0       # Eve's intercept rate
    noise_enabled       = False     # Add noise?
    noise_model         = "depolarizing"
    depolar_prob        = 0.01      # noise probability
    sample_fraction     = 0.10      # % of key for QBER check
    seed                = 42        # reproducible runs
```

---

## 🗂️ File Structure

```
BB84_QKD/
├── app.py                  ← Main Streamlit interface (NEW)
├── bb84_core.py           ← Quantum simulation logic
├── bb84_runner.py         ← Orchestrator & scenarios
├── bb84_plots.py          ← Matplotlib visualizations
├── bb84_attacks.py        ← Eve's attack models
├── bb84_noise.py          ← Channel noise models
├── bb84config.py          ← Configuration dataclasses
└── BB84_Phase1.ipynb      ← Jupyter notebook
```

---

## 📱 Browser Tips

- Works on **Chrome, Firefox, Safari, Edge**
- Mobile responsive (tablets/phones supported)
- Use Ctrl+Shift+R to hard-refresh if plots don't update
- Plots are interactive (hover, zoom, download)

---

## 🚀 Tips for Research

1. **Compare Scenarios** page shows impact of:
   - Depolarizing noise on QBER
   - Eve's intercept rate on security
   - Combined noise + Eve attacks

2. **Eve Analysis** page helps determine:
   - Minimum Eve activity needed for detection
   - Protocol robustness thresholds
   - Noise tolerance of security

3. Save results by:
   - Screenshot (built-in browser tool)
   - Export data to CSV (from table)
   - Run with same seed for reproducibility

---

## 📝 Troubleshooting

**"Module not found" errors:**
```bash
pip install -r requirements.txt
```

**Plots not showing:**
- Clear cache: `streamlit cache clear`
- Hard refresh browser: Ctrl+Shift+R

**Performance slow:**
- Reduce number of qubits in config
- Use fewer points in Eve Analysis sweep
- Run simulations with larger batches

---

## 🔧 Customize the Interface

Edit `app.py` to:
- Change color scheme
- Add new analysis plots
- Modify parameter ranges
- Add custom scenarios

Streamlit docs: [streamlit.io](https://streamlit.io)

Plotly docs: [plotly.com/python](https://plotly.com/python)
