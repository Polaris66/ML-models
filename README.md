# ML_assignment_1

Code for the First Assignment of the Machine Learning Course (2023 - Sem 2) Assignments.

Team Members:
1) Dhruv Arora
2) Jatin Jindal
3) Pradyumna Malladi

### Labels

Malignant = 1

Benign = -1 for perceptron, FLDA

Benign = 0 for logistic

# Perceptron Algorithm

After 10000 epochs, data gives 90-92% accuracy, 90-94% precision and 85-90% recall when tested using average of 10 different train-test splits.

It is expected that even after further iterations Perceptron will not outperform normalized data.

After normalizing data, the algorithm converges under 10000 epochs giving 94% accuracy, 92% precision and 94% recall.

Rearranging the feature order does not affect the result. The weight vector appears similarly rearranged.

# Fischer's Linear Discriminant Analysis

Fischer's LDA does not require training and directly gives 95% accuracy, 95% precision and 91% recall on normalized and imputed data.

Rearranging the feature order does not affect the result. The weight vector appears similarly rearranged.

# Logistic Regression

After running for a 1000 epochs it gives maximum of 97% accuracy. It is expected to go even higher with more epochs.

## Batch Gradient Descent

Batch Gradient Descent shows a 74-89% accuracy, 63-89% precision and 76-81% recall with un-normalized data all decreasing with decrease in learning rate.

With normalized data, it gives 88-93% accuracy, 82-92% precision and 89-95% recall.

## Stochastic Gradient Descent

Stochastic Gradient Descent shows a 89-91% accuracy, 88-90% precision, 81-88% recall, all except precision decreasing with decrease in learning rate.

With normalized data, it gives 95-97% accuracy, 91-97% precision and 89-96% recall.

## Mini-Batch Gradient Descent

It shows 86-89% accuracy, 84-88% precision, 83-85% recall with raw data.

Shows 95-97% accuracy, 92-97% precision and 94-95% recall with normalized data.

## Comparative Study

After 10000 epochs (except for Fischer) following is the table of observations. (Normalized Data Only)

| Models     | Accuracy          | Precision         | Recall            |
| :--------- | :---------------- | :---------------- | :---------------- |
| Perceptron | 0.943010752688172 | 0.916875661563711 | 0.937355374697079 |
| Fischer    | 0.950531914893617 | 0.952921328366989 | 0.914998930095255 |
| Batch      | 0.968085106382979 | 1                 | 0.916666666666667 |
| Stochastic | 0.968085106382979 | 0.958333333333333 | 0.958333333333333 |
| Mini Batch | 0.968085106382979 | 0.958333333333333 | 0.958333333333333 |

Logistic Regression Metrics have been run for 10000 epochs for only a specific learning rate and threshold and may vary based on others. However, it clearly outperforms both Fischer and Perceptron.
