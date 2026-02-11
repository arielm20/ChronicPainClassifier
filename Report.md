# Overview
The analysis looked at EEG frequency band power patterns that distinguish three different chronic pain conditions: Fibromyalgia (FM), Chronic Back Pain (CBP) and Nocioceptive Chronic Pain (NCCP).
## Methodology and Results
### Pre-processing
To clean up the raw EEG data, I applied a Butterworth bandpass filter (1-45Hz), and an independent component analysis (ICA). Then, both the regional and global relative band powers were calculated for the Delta (1-3Hz), Theta (3-7Hz), Alpha (7-13Hz), and Beta (13-30Hz) bands. Relative band power is represented between 0 and 1, measuring the proportion of energy within a specific frequency band relative to the total spectrum.
### Feature selection
Mutual information (MI) analysis was used to identify the most discriminative features. MI is a statistical method that quantifies the shared entropy between two random variables, and in the case of EEG this method is prefered over other statistical methods such as ANOVA due to MI's ability to quantify both linear and non-linear dependencies and because it does not assume normal distribution. MI scores range between 0 and 1, where 0 indicates independence and 1 shows full dependence.
#### Selected features
|Feature Band      |     Frequency |     Region |     MI Score |
| :----------:     | :-----------: | :--------: | :----------: |
| Delta            | 1-3 Hz        | Global     | 0.492        |
| Theta            | 3-7 Hz        | Global     | 0.438        |
| Beta             | 13-30 Hz      | Global     | 0.280        |
| Alpha            | 7-13 Hz       | Global     | 0.097        |

The MI scores show a clear pattern in increasing frequency band importance in chronic pain diagnosis. The delta and theta frequency bands show the highest associations, which aligns with previous literature showing these bands are implicated in the thalamocortical dysrhythmia seen in chronic pain. The dominance of global relative power across these bands suggests that chronic pain may manifest as a widespread shift in neural oscillations rather than a localized abnormality. Either way, chronic pain detection may benefit from considering multiple frequency bands.
### Model Development
Optuna was used to evaluate multiple classifier types as well as their hyperparameter configurations
- XGBoost
- CatBoost
- Gradient Boosting
- Random Forest
- Support Vector Machines

XGBoost emerged as the best-performing model, with the following hyperparameter configuration:

|Parameter         | Value     |
| :----------:     | :-------: |
| Learning Rate    | 0.012     | 
| Estimators       | 817       | 
| Max Depth        | 5         | 
| Min Child Weight | 7         |
| Gamma            | 1.99e-10  |

The multi-class classifier achieved the following results:
1. Chronic Back Pain
    - Sensitivity: 70%
    - Specificity: 84%
2. Nocioceptive Chronic Pain
    - Sensitivity: 67%
    - Specificity: 85%
3. Fibromyalgia
    - Sensitivity: 100%
    - Specificity: 98%

### Key takeaways
This analysis showed that chronic pain conditions can be distinguished by distinct spectral fingerprints, notably the global dominance of the delta and theta frequencies. The XGBoost model proved robust in achieving near-perfect classification of Fibromyalgia; however, the overlap in features between CBP and NCCP suggests that additional features (such as coherence or functional connectivity) should be analyzed. Clinically, these results suggest that while EEG frequency band-based classifiers are a powerful tool for identifying some chronic pain conditions, their utility is not universal.