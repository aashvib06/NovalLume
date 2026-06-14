# NILM Appliance Disaggregation using UK-DALE

## Dataset Selection

I selected **UK-DALE** because it is one of the most widely used residential NILM datasets and provides both aggregate household power consumption and appliance-level ground truth measurements. House 1 was used because it contains the longest recording period and the largest number of monitored appliances.

The dataset has a sampling interval of approximately **6 seconds**, which is sufficient for low-frequency NILM and is commonly used in published Seq2Point NILM research.

---

## Appliances Chosen

Three appliances were selected from different usage categories:

| Appliance | Reason                               |
| --------- | ------------------------------------ |
| Kettle    | High-power, short-duration appliance |
| Fridge    | Cyclic compressor-based appliance    |
| Microwave | Medium-power, short burst appliance  |

This provides a mix of easy, moderate, and difficult disaggregation scenarios.

---

## Preprocessing Pipeline

### 1. Data Extraction

Power readings were extracted from:

```text
Aggregate (Meter 1)
Kettle (Meter 10)
Fridge (Meter 12)
Microwave (Meter 13)
```

using the UK-DALE HDF5 structure.

### 2. Timestamp Alignment

A major issue encountered was that appliance meters were not perfectly synchronized with the aggregate meter. An initial inner join resulted in an empty dataset because very few timestamps matched exactly across all channels.

This was resolved by:

* Using an outer join
* Sorting timestamps
* Forward-filling and backward-filling missing values
* Resampling to a consistent 6-second interval

### 3. Cleaning

The following cleaning steps were applied:

* Removed negative readings
* Removed invalid values (NaN, Inf)
* Filled missing values
* Standardized all channels onto the same timeline

---

## Feature Engineering

Sliding windows were generated from aggregate power.

For each window, the following features were extracted:

```text
Mean Power
Standard Deviation
Minimum Power
Maximum Power
Range
Energy
Mean Power Change
Power Change Standard Deviation
Maximum Rise
Maximum Drop
```

These features capture:

* Power magnitude
* Variability
* Energy consumption
* Load transitions

which are important characteristics for NILM.

---

## Initial Baseline Model

A **Random Forest Regressor** was used as the baseline NILM model.

Input:

```text
Aggregate window features
```

Output:

```text
Predicted appliance power
```

The model was trained separately for each appliance.

---

## Challenges Encountered

### 1. Empty Dataset After Alignment

Initially, all channels were merged using an inner join.

Because appliance timestamps were not perfectly aligned, this produced:

```text
0 rows
```

and no training data.

Switching to an outer join followed by interpolation and resampling solved this issue.

---

### 2. Severe Class Imbalance

Most appliances are OFF for the majority of the time.

For example:

* Kettle operates for only a few minutes per day.
* Microwave usage occurs in short bursts.

As a result, the model sees many more OFF samples than ON samples.

This explains why:

* Fridge achieved strong performance.
* Kettle achieved reasonable performance.
* Microwave was the most difficult appliance.

---

### 3. Dataset Size

The cleaned dataset contained tens of millions of samples.

Generating windows across the entire dataset was computationally expensive, so a representative subset was used for initial experiments.

---

## Results

### Kettle

The model successfully identified high-power kettle events.

Performance was strong because kettle activations are very distinctive and typically exceed 2000W.

---

### Fridge

The fridge produced the best overall results.

Reasons:

* Frequent cycling behaviour
* Repeating operating pattern
* Large number of training examples

This allowed the model to learn fridge behaviour effectively.

---

### Microwave

Microwave performance was lower than kettle and fridge.

Reasons:

* Short activation periods
* Fewer training examples
* Greater overlap with other household loads

This makes microwave disaggregation more challenging.

---

## Visual Analysis

Several visualizations were generated:

### Aggregate vs Predicted Appliance Power

Shows how predicted appliance power follows appliance activation events within the aggregate signal.

### Aggregate Reconstruction

The predicted appliance powers were summed and compared against the aggregate signal to evaluate overall reconstruction quality.

### Confusion Matrices

Used to evaluate ON/OFF state detection performance.

### Loss Curves

Used to monitor training convergence.

### Residual Analysis

Used to identify prediction errors and difficult operating regions.

---

## Final Pipeline

```text
UK-DALE Dataset
        ↓
Data Extraction
        ↓
Timestamp Alignment
        ↓
Cleaning & Resampling
        ↓
Feature Engineering
        ↓
Random Forest Baseline
        ↓
Per-Appliance Evaluation
        ↓
Visualization & Error Analysis
```

## Future Improvements

* CNN Sequence-to-Point (Seq2Point)
* Longer sliding windows
* Better handling of class imbalance
* Window-based temporal learning
* Iterative subtraction NILM
* Full-dataset training instead of subsets

The current implementation successfully establishes a complete NILM pipeline, from raw UK-DALE data extraction through preprocessing, feature engineering, appliance disaggregation, and evaluation.
