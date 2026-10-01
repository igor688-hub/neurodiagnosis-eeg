# Data audit and solution log

What we found in the data, what we decided because of it, and how each step ended. Every protocol was written down before it was run.

## 1. Data

| Group | Subjects | Recordings | Age |
| --- | ---: | --- | --- |
| PTSD | 25 | rest and five Schulte tables | unknown (18–45 by the task description, men) |
| Controls, original release | 67 | rest and five Schulte tables | 19–58 |
| Controls, supplementary batch | 40 | rest only | 17–41 |
| Controls aged 65+ | 23 | rest and the first table | 65–70 |
| Somatoform disorders | 74 | rest and five Schulte tables | unknown (18–45, both sexes) |

- The reference electrode is on the left earlobe. T3 is the closest lead to it, so part of the T3/T4 asymmetry comes from the montage.
- Some recordings are byte-identical copies stored under different IDs. Merging them gives 229 subjects in 221 independent groups, and every data split is made by group.
- Files whose content appears under both a rest name and a task name are excluded by the same rule in training and in inference.

## 2. The main problem: export format follows the label

Each subject was exported in one of three EDF formats.

| Format | Amplitude step | PTSD | Controls | Somatoform |
| --- | --- | ---: | --- | ---: |
| A | 1 µV | 0 | 20 | 74 |
| B | 0.06 µV | 25 | 2, plus the supplementary batch and the 65+ group | 0 |
| C | about 0.01 µV | 0 | 45 | 0 |

- The rule "format B means PTSD" separates the original PTSD and control groups almost perfectly (AUC 0.985) without looking at the signal.
- Format C looks unlike typical EEG: symmetric channels are uncorrelated and there is no occipital alpha. We could not find the reason and kept the records.
- The supplementary batch has copied channels in some files, an O2 channel that behaves like a frontal one and a different high-frequency spectrum. These are signs of a different recording setup.
- Age is mixed up with format: most controls older than 45 are in format C.

Decisions that follow: the model never sees file headers or paths; features use only frequencies below 40 Hz and do not depend on channel gain; O2 and copied channels are excluded; a model trained on export metadata alone serves as a required control; results are always reported per group.

## 3. Preprocessing and validation

Signals are resampled to 125 Hz, low-pass filtered at 40 Hz and cut into 4-second windows. Windows with a flat signal, clipping, very large amplitude or a copied channel are rejected per channel, with the same thresholds for everyone. The spectrum of each channel is averaged over the remaining windows. Recording quality differs between cohorts (patients lose more windows), so quality measures are not given to the model.

Validation is leave-one-group-out. Inside each training fold, a group-stratified 5-fold cross-validation chooses the feature set, the regularisation, the negative class and the calibration. Outer predictions are never used for any choice. Confidence intervals come from a bootstrap over groups.

## 4. Protocols

| Protocol | What was tested | Result | Conclusion |
| --- | --- | --- | --- |
| 1 | rest and task features, all channels | high overall AUC, chance level against control A, and a metadata-only model did as well | the headline AUC came from format C |
| 2 | rest only, no O2, spectral slope, supplementary batch added | separation within format B relied on T3 and the spectrum above 20 Hz | these look like batch features |
| 3 | only the most robust rest features (O1, Fp1, Fp2, 4–20 Hz) | chance level against comparable controls | an honest negative result |
| 4 | Schulte behaviour and task EEG | solve time was the most robust signal, task EEG added nothing | older adults solve slower than patients |
| 5 | first-table solve time plus rest EEG | modest gain | behaviour left out of the final model, since the task is EEG diagnostics |
| 6 | rest EEG only, all subjects, selection by the official scoring formula | clear separation from controls under 65 | features are not removed on suspicion, confounders are tested separately |
| 7 | protocol 6 plus frontal alpha asymmetry, balanced calibration | sensitivity restored | asymmetry added nothing |
| 8 | protocol 7 plus extra weight for the specificity groups and an equal-error threshold | better AUC and specificity | submitted |

## 5. Submitted model (protocol 8)

The model uses 23 rest features from O1, Fp1, Fp2, T3 and T4: relative theta, alpha and beta power, alpha peak frequency and height on O1, the slope of the aperiodic spectrum and frontal alpha asymmetry. The classifier is a logistic regression with an L2 penalty and balanced class weights. Controls aged 65+ and somatoform patients may receive extra weight, decided inside each fold. Platt calibration places the 0.5 threshold where sensitivity on PTSD equals specificity on these two groups. The same feature set was chosen in every fold.

| Metric (out of sample) | Value [95% CI] |
| --- | --- |
| AUC, PTSD vs controls under 65 | 0.86 [0.80; 0.92] |
| AUC, PTSD vs controls without format C | 0.77 [0.66; 0.86] |
| Sensitivity at 0.5 | 0.84 [0.69; 0.96] |
| Specificity, controls under 65 | 0.72 [0.64; 0.80] |
| Specificity, controls aged 65+ | 0.87 [0.74; 1.00] |
| Specificity, somatoform | 0.61 [0.50; 0.72] |

Protocol 8 replaced protocol 7 under a rule set in advance: specificity on the 65+ and somatoform groups had to improve while sensitivity stayed at 0.6 or above and AUC did not drop. All conditions held.

Specificity on young controls varies by batch. It is almost perfect on format C and about one half on control A.

## 6. Checks after model selection

- Export metadata alone and recording quality alone do not reproduce the result.
- Within format B the model still separates PTSD from controls (AUC 0.84). Shuffling labels inside format B and repeating the whole nested procedure 60 times never reached that value (p = 0.016).
- When a control group is removed from the whole procedure and then predicted as new data, AUC drops by about 0.1. Older adults are still separated best.
- Within a batch the features do not predict age, and P does not follow age.
- P does not follow Schulte solve time.
- Without T3 and T4, separation from young controls is close to chance, while separation from older adults remains.
- The two features reliably shifted in PTSD, a flatter spectrum and more beta at T4, are shifted the same way in somatoform patients. No feature is specific to PTSD.

## 7. Bonus: metronome

The recordings contain a metronome but no stimulus markers. We fold each recording into 4-second cycles of eight beats and separate the beat and deviant responses by frequency. The timing is searched on one half of the cycles and tested on the other, and a random-shift control repeats the whole search.

- A response to the beats exists at rest and is largest at Fp1 and Fp2 (p = 0.0005). It is not found during the Schulte task.
- On the open ERP CORE dataset our loader reproduces the published mismatch negativity. Simulating the headset's 2 Hz hardware high-pass filter turns it into a three-phase wave at Fp.
- In a hybrid simulation the method detects a deviant response only if it is larger than a typical MMN.
- The single pre-registered confirmatory test missed its threshold (p = 0.048 against 0.025). The deviant response (MMN/P3a) is not confirmed.

## 8. Limitations

1. Separation from young controls of the same format relies on T3 and T4, the leads most sensitive to recording conditions. Physiology and batch cannot be told apart there.
2. Somatoform patients receive lower P than healthy format A controls, so they are not separated by a clinical profile. Sex is a possible confounder: all PTSD patients are men, and sex is not recorded.
3. The profile shared by PTSD and somatoform patients, more beta and a flatter spectrum at T4, also matches the EEG effect of benzodiazepines. Medication is unknown.
4. Cohorts differ beyond the diagnosis: civilian volunteers against a clinical cohort with combat trauma, with unknown medication, head injury and sleep.
5. All estimates come from the development data. The independent estimate is the organisers' closed test.

Clinical interpretation: [CLINICAL_INTERPRETATION.md](CLINICAL_INTERPRETATION.md).
