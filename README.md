## Mixup Asymmetric Tri-training for Heartbeat Classification Under Domain Shift <br>
In this paper, we propose a novel Mixup Asymmetric Tri-training (MIAT) method to improve the generalization ability of heartbeat classifiers
under domain shift scenarios.

## Main requirements

  * **torch == 1.0.0**
  * **Python == 3.5**
  * **wfdb == 1.2.2**

## Task

Classify ECG heartbeats into 5 classes: N, S, V, F, Q


## Usage
```
# (1) Download data
Download [MIT-BIH Arrhythmia Database (MITDB)] (https://www.physionet.org/content/mitdb/1.0.0/)
Download [MIT-BIH Supraventricular Arrhythmia Database (SVDB)] (https://www.physionet.org/content/svdb/1.0.0/)

# (2) preprocess to get npy
python preprocess.py

# (3) train
python test_mitdb.py

```

