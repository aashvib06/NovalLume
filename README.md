# NILM Appliance Disaggregation using UK-DALE

## Dataset Selection

The UK-DALE dataset was selected as the primary NILM dataset. UK-DALE is one of the most widely used residential energy datasets and contains aggregate household power measurements along with appliance-level ground truth labels.

The dataset uses approximately 6-second sampling, which is sufficient for low-frequency appliance disaggregation. Although higher-frequency datasets exist, switching datasets would not have solved the main issues encountered during development. Most published NILM baselines, including the original Seq2Point work, operate using aggregate active power only rather than voltage/current waveforms, harmonics, THD, or FFT features.

Therefore, UK-DALE remained a suitable choice for building a strong NILM baseline.

---

## Appliances Selected

Three appliances were selected:

### Kettle

* High power appliance (~2000W+)
* Short operating duration
* Easy NILM benchmark
* Used as a sanity check

### Fridge

* Cyclic compressor-based appliance
* Repeating operating pattern
* Good test of temporal behavior

### Microwave

* Medium-power appliance
* Short burst activations
* More difficult to disaggregate due to overlap with other loads

Together, these appliances represent three common NILM categories:

* On/Off resistive load (Kettle)
* Cyclic compressor load (Fridge)
* Short-duration burst load (Microwave)

---

## Major Issues Encountered

### 1. Timestamp Alignment Failure

Initially, appliance channels were merged using exact timestamp matching.

This produced an almost empty dataset because appliance meters and aggregate meters are not perfectly synchronized in UK-DALE.

The solution was:

* Outer join across channels
* Forward fill / backward fill missing values
* Resample all signals to a common 6-second timeline

---

### 2. Dataset Imbalance

A much more significant issue was class imbalance.

Most appliances are OFF for the vast majority of the dataset:

* Kettle is OFF >99% of the time
* Microwave is OFF most of the time
* Fridge is active more frequently but still highly imbalanced

As a result, naive models can achieve low error simply by predicting values close to zero.

This became the dominant challenge of the project.

---

### 3. Incorrect Data Splitting

An early mistake was using:

```python
df.iloc[:500000]
```

to create training data.

The problem is that a contiguous chunk of data may contain little or no appliance activity.

This produced degenerate targets where some appliances appeared almost constant.

The correct approach is:

* Use the full aligned dataset
* Generate windows across the entire timeline
* Split by calendar periods (weeks/months)
* Ensure both active and inactive appliance periods appear in all splits

---

### 4. Random Forest Failure

The Random Forest model often appeared to predict nearly constant values.

This was not a software bug.

It was the expected behavior of a regression model minimizing MSE on an extremely imbalanced target distribution.

Because most samples correspond to appliance OFF states, predicting values near zero minimizes overall loss.

This highlighted the need for temporal sequence models rather than simple feature-based regressors.

---

## Final Preprocessing Pipeline

### Data Extraction

Extract:

* Aggregate (Meter 1)
* Kettle (Meter 10)
* Fridge (Meter 12)
* Microwave (Meter 13)

from UK-DALE House 1.

### Cleaning

* Remove invalid values
* Remove negative readings
* Sort timestamps
* Outer join appliance channels
* Forward fill missing values
* Backward fill remaining gaps
* Resample to 6-second intervals

### Outlier Handling

Extreme aggregate spikes above realistic household consumption levels are removed as probable sensor errors.

### Normalization

Before CNN training:

* Inputs are normalized using training-set statistics only
* Outputs are normalized using appliance-specific statistics

This stabilizes optimization because appliance powers range from a few watts to several thousand watts.

---

## Window Generation

Instead of individual timestamps, sliding windows are generated.

Window lengths between:

* 99 samples (~10 minutes)
* 599 samples (~1 hour)

were considered.

Longer windows provide sufficient context to capture:

* Kettle activations
* Fridge compressor cycles
* Microwave bursts

Windows are generated across the full timeline rather than from a small contiguous subset.

To avoid memory issues, windows are generated lazily using dataset generators instead of storing all windows in RAM.

---

## Final Model Choice

### Random Forest

Used only as a baseline.

Advantages:

* Fast
* Easy to interpret

Limitations:

* No temporal modeling
* Performs poorly on short transient appliances

---

### CNN Sequence-to-Point (Seq2Point)

Selected as the primary NILM model.

Reasons:

* Standard NILM architecture
* Proven on UK-DALE
* Captures temporal context
* Handles appliance events significantly better than classical regressors

One separate model is trained per appliance:

* Kettle model
* Fridge model
* Microwave model

This follows standard NILM practice and simplifies debugging.

---

## Seq2Point Architecture

Input:

599-sample aggregate power window

Network:

* Conv1D(30 filters, kernel=10)
* Conv1D(30 filters, kernel=8)
* Conv1D(40 filters, kernel=6)
* Conv1D(50 filters, kernel=5)
* Conv1D(50 filters, kernel=5)
* Flatten
* Dense(1024)
* Dropout(0.2)
* Dense(1)

Output:

Predicted appliance power at the center timestamp of the window.

This architecture contains approximately one million parameters and is based on the original Seq2Point NILM architecture.

---

## Evaluation Metrics

The following metrics were used:

### MAE (Mean Absolute Error)

Primary regression metric measured in watts.

### SAE (Signal Aggregate Error)

Measures total energy estimation error.

### Precision

Measures correctness of appliance ON detections.

### Recall

Measures how many true appliance activations were detected.

### F1 Score

Balances precision and recall and is particularly important for highly imbalanced appliances.

### R²

Reported only with caution because it can be misleading for NILM datasets with highly imbalanced targets.

---

## Visualizations

The following visualizations were generated:

### Time-Series Overlay

Ground truth appliance power versus predicted appliance power.

### Aggregate Reconstruction

True aggregate power versus reconstructed aggregate power from predicted appliance outputs.

### Confusion Matrices

ON/OFF detection performance.

### Power Histograms

Distribution of appliance power values.

### Training and Validation Loss Curves

Training convergence analysis.

### Residual Analysis

Prediction error visualization.

---

## Results Interpretation

### Kettle

Strong performance.

The kettle has a very distinctive high-power signature and is relatively easy to detect.

### Fridge

Best overall performance.

The repeating compressor cycle provides many training examples and strong temporal patterns.

### Microwave

Most difficult appliance.

Microwave events are short, infrequent, and often overlap with other loads, making disaggregation harder.

---

## Future Work

The following were considered but not implemented in the baseline:

* Seq2Seq architectures
* Iterative subtraction methods
* Multi-task NILM models
* High-frequency waveform features
* Real-time streaming inference

The current pipeline focuses on establishing a strong, reproducible Seq2Point NILM baseline using UK-DALE.
