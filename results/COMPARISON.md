# Person 3 – CNN Optimization Results

## Completed genetic-algorithm search

- Evaluations completed: **67**
- Best validation macro-F1: **0.8220**
- Best fitness: **0.8162**
- Best CNN settings: 4 convolution blocks, 16 starting filters, 5×5 kernel, 0.2 dropout, 128 dense units, learning rate 0.0001, batch size 8, adam optimizer, and augmentation multiplier 1.0.

## Partial random-search control

- Planned evaluations: **67**
- Recovered evaluations: **28**
- Best validation macro-F1 among recovered trials: **0.7278**
- Best fitness among recovered trials: **0.7261**
- Best recovered settings: 3 convolution blocks, 16 starting filters, 5×5 kernel, 0.1 dropout, 256 dense units, learning rate 0.002154434690031882, batch size 8, sgd optimizer, and augmentation multiplier 0.5.

## Interpretation

The genetic algorithm found a model with validation macro-F1 **0.8220**, which was **0.0942** higher than the best model found in the 28 recovered random trials. This suggests that the genetic algorithm searched the CNN settings effectively. However, the random-search run was interrupted before reaching its planned 67 trials, so this should be described as a **partial exploratory comparison**, not a final equal-budget comparison. The held-out test set was not used during either search.

## Files

- `ga_search_recovered.json`: all 67 completed GA evaluations.
- `random_search_partial_28.json`: all 28 random-search evaluations preserved from the interrupted Colab run.
- `Untitled0_recovery_source.ipynb`: notebook containing the original output used for recovery.
