
# UK-DALE High-Frequency NILM Preprocessing and Feature Engineering Pipeline

## 1. Dataset Selection

### Selected Dataset

**UK-DALE (UK Domestic Appliance-Level Electricity Dataset)**

### Source

https://data.ceda.ac.uk/edc/d1/7d78f943-f9fe-413b-af52-1816f9d968b0/data/version_0

### Rationale

The objective of this project is to develop a preprocessing and feature engineering pipeline for Non-Intrusive Load Monitoring (NILM) using high-resolution electrical measurements. UK-DALE was selected because it provides one of the most comprehensive residential energy datasets available, containing synchronized aggregate mains measurements and appliance-level ground truth recordings.

Unlike conventional NILM datasets that provide only active power readings at low sampling rates, the high-frequency version of UK-DALE includes detailed voltage and current waveform recordings. This enables the extraction of advanced electrical signatures such as harmonic content, waveform distortion metrics, power quality indicators, and voltage-current trajectory characteristics. These signatures are highly informative for distinguishing appliances with similar power consumption patterns but different electrical behaviours.

The dataset is therefore well suited for both traditional feature-based NILM approaches and modern deep learning architectures such as CNNs and LSTMs.

---

## 2. Sampling Frequency Selection

### Sampling Rate

**16 kHz**

### Justification

The preprocessing pipeline is designed around the native 16 kHz sampling frequency of the UK-DALE high-frequency recordings.

Residential mains electricity in the UK operates at a fundamental frequency of 50 Hz. A high sampling rate is necessary to accurately capture waveform shape, phase relationships, and harmonic distortion produced by household appliances.

Using 16,000 samples per second provides several advantages:

* Accurate representation of the 50 Hz voltage and current waveforms.
* Reliable extraction of harmonic components beyond the fundamental frequency.
* Improved estimation of phase shifts and power factor.
* Capture of transient appliance behaviour during switching events.
* Support for detailed V-I trajectory analysis.
* Sufficient frequency resolution for FFT-based feature extraction.

The selected sampling frequency significantly exceeds the minimum Nyquist requirement, ensuring that important electrical characteristics are preserved during preprocessing.

---

## 3. Data Cleaning and Preprocessing

Before feature extraction, the raw voltage and current signals undergo several preprocessing operations to improve signal quality and ensure consistency across windows.

### Missing Value Handling

Any missing or corrupted samples are replaced using numerical interpolation or zero-substitution depending on the severity of the gap. This prevents invalid values from propagating into downstream feature calculations.

### DC Offset Removal

Measurement hardware often introduces small DC biases into waveform recordings. The mean value of each voltage and current window is removed to ensure the signals are centred around zero.

This step improves FFT quality and prevents artificial low-frequency components from appearing in the spectral representation.

### Noise and Outlier Mitigation

Extreme outlier values caused by sensor glitches or acquisition errors are clipped to a physically reasonable range. This prevents individual corrupted samples from disproportionately influencing RMS and harmonic calculations.

### Window Segmentation

The continuous waveform stream is divided into fixed-length windows.

Configuration:

* Sampling Frequency: 16 kHz
* Window Length: 16,000 samples
* Window Duration: 1 second
* Overlap: 50%

Windowing converts the continuous signal into manageable segments suitable for feature extraction and deep learning models.

### Signal Normalization

For machine learning compatibility, cleaned voltage and current waveforms can optionally be standardized using training-set statistics. This improves convergence behaviour during CNN and LSTM training.

---

## 4. Feature Engineering

A comprehensive set of electrical, spectral, statistical, and waveform-based features is extracted from every window.

### A. Electrical Features

These features characterize the overall power consumption behaviour of the appliance.

#### Voltage RMS (Vrms)

Measures the effective voltage magnitude.

#### Current RMS (Irms)

Measures the effective current drawn by the load.

#### Active Power (P)

Represents the real electrical power consumed.

#### Apparent Power (S)

Computed as:

S = Vrms × Irms

#### Reactive Power (Q)

Calculated using the power triangle relationship.

#### Power Factor (PF)

Represents the ratio between active and apparent power and provides insight into the phase relationship between voltage and current.

---

### B. Harmonic Features

To capture frequency-domain appliance signatures, a Fast Fourier Transform (FFT) is applied to the current waveform.

The following harmonic magnitudes are extracted:

* Fundamental Harmonic (50 Hz)
* 3rd Harmonic (150 Hz)
* 5th Harmonic (250 Hz)
* 7th Harmonic (350 Hz)
* 9th Harmonic (450 Hz)
* 11th Harmonic (550 Hz)
* 13th Harmonic (650 Hz)

Different appliances exhibit distinct harmonic fingerprints due to their internal electrical characteristics.

---

### C. Total Harmonic Distortion (THD)

THD quantifies the amount of harmonic distortion present relative to the fundamental frequency component.

A low THD value typically corresponds to resistive loads, whereas electronic devices with switching power supplies often exhibit higher THD values.

---

### D. Spectral Features

Additional frequency-domain descriptors are extracted to characterize energy distribution within the signal.

#### Spectral Centroid

Indicates the centre of mass of the frequency spectrum.

#### Spectral Entropy

Measures the complexity and irregularity of the spectral content.

---

### E. Waveform Shape Features

These features describe the geometric characteristics of the voltage and current waveforms.

#### Crest Factor

Ratio of peak value to RMS value.

#### Form Factor

Ratio of RMS value to mean absolute value.

#### Peak Voltage

Maximum voltage magnitude within the window.

#### Peak Current

Maximum current magnitude within the window.

---

### F. Statistical Features

Statistical descriptors are computed independently for both voltage and current signals.

Features include:

* Mean
* Standard Deviation
* Variance
* Skewness
* Kurtosis

These features capture distributional properties and signal shape characteristics.

---

### G. V-I Trajectory Features

Voltage-current trajectory analysis is widely used in NILM because it captures appliance behaviour independent of absolute power consumption.

Extracted features include:

#### Loop Area

Represents the enclosed area of the V-I trajectory.

#### Voltage Span

Difference between maximum and minimum voltage.

#### Mean Slope

Average slope of the V-I curve.

These features often provide strong appliance discrimination capability.

---

## 5. Output Generation

The preprocessing pipeline generates two complementary outputs.

### Feature Dataset

A structured feature table is generated for each processed window and stored in columnar format.

Contents include:

* Electrical Features
* Harmonic Features
* Spectral Features
* Statistical Features
* V-I Trajectory Features
* Ground Truth Labels

This dataset is suitable for classical machine learning algorithms.

---

### Waveform Dataset

Cleaned voltage and current windows are stored as tensors with shape:

(N, 2, Window_Size)

where:

* Channel 1 = Voltage
* Channel 2 = Current

These waveform representations are designed for direct use with CNN and LSTM architectures.

---

## 6. Final Feature Set

The final feature vector consists of:

* Vrms
* Irms
* Active Power
* Apparent Power
* Reactive Power
* Power Factor
* Harmonics H1–H13
* Total Harmonic Distortion
* Spectral Centroid
* Spectral Entropy
* Voltage Crest Factor
* Current Crest Factor
* Voltage Form Factor
* Current Form Factor
* Voltage Statistics
* Current Statistics
* V-I Loop Area
* V-I Span
* V-I Mean Slope

This combination captures both steady-state and frequency-domain appliance characteristics, providing a rich representation suitable for NILM applications.
