# NILM Appliance Disaggregation using UK-DALE

## Dataset

* Dataset: UK-DALE (House 1)
* Sampling Rate: 6 seconds
* Appliances Used:

  * Kettle
  * Fridge
  * Microwave

UK-DALE was retained because it is a standard NILM benchmark and active power measurements are sufficient for building a strong baseline disaggregation model.

---

## Final Pipeline

1. Extract aggregate and appliance-level power data from UK-DALE.
2. Align appliance and aggregate timestamps.
3. Apply:

   * Outer join
   * Forward fill / backward fill
   * 6-second resampling
4. Remove invalid and negative readings.
5. Normalize data using training-set statistics.
6. Generate sliding windows (Seq2Point).
7. Train one CNN Seq2Point model per appliance.
8. Evaluate using MAE, SAE, Precision, Recall and F1 Score.

---

## Issues Faced

### Timestamp Alignment

* Appliance and aggregate channels were not perfectly synchronized.
* Initial exact timestamp matching produced almost empty datasets.
* Fixed using outer joins and resampling.

### Class Imbalance

* Kettle and microwave are OFF most of the time.
* Models can easily learn to predict near-zero values.
* This made appliance detection significantly harder.

### Incorrect Data Splitting

* Early experiments used contiguous chunks of data.
* Some chunks contained little appliance activity.
* Fixed by generating windows across the full timeline and splitting by time periods.

### Random Forest Baseline

* Random Forest frequently predicted near-constant values.
* This was due to severe target imbalance and lack of temporal context.
* Motivated the move to Seq2Point CNNs.

---

## Final Model

### Seq2Point CNN

Input:

* Aggregate power window (599 samples)

Output:

* Appliance power at the center timestamp

Separate models were trained for:

* Kettle
* Fridge
* Microwave

---

## Final Results

| Appliance | MAE (W) | SAE   | Precision | Recall | F1    |
| --------- | ------- | ----- | --------- | ------ | ----- |
| Kettle    | 11.77   | 0.231 | 0.976     | 0.675  | 0.798 |
| Fridge    | 15.34   | 0.064 | 0.899     | 0.887  | 0.893 |
| Microwave | 21.92   | 0.460 | 0.373     | 0.927  | 0.532 |

---

## Observations

* Fridge achieved the best overall performance due to its regular cyclic behaviour.
* Kettle was detected reliably because of its distinct high-power signature.
* Microwave was the most difficult appliance because of short, infrequent activations and overlap with other household loads.
* Seq2Point CNN significantly outperformed the initial Random Forest baseline by learning temporal appliance patterns directly from aggregate power windows.

---

## Visualizations Generated

* Ground Truth vs Predicted Appliance Power
* Aggregate vs Reconstructed Aggregate Power
* Confusion Matrices
* Training/Validation Loss Curves
* Residual Error Analysis
* Appliance Power Histograms

These plots were used to validate model behaviour and identify appliance-specific prediction errors.
