## Mixup Asymmetric Tri-training for Heartbeat Classification Under Domain Shift <br>
In this paper, we propose a novel Mixup Asymmetric Tri-training (MIAT) method to improve the generalization ability of heartbeat classifiers
under domain shift scenarios.

## Main requirements

  * **Torch == 1.0.0**
  * **Python == 3.5**
  * **WFDB == 1.2.2**

## Task
Classify ECG heartbeats into 5 classes: N, S, V, F, Q

## Dataset
Download [MIT-BIH Arrhythmia Database (MITDB) ](https://www.physionet.org/content/mitdb/1.0.0/) <br>
Download [MIT-BIH Supraventricular Arrhythmia Database (SVDB)](https://www.physionet.org/content/svdb/1.0.0/)


## Usage
```
(1) Cut the continuous ECG signals into heartbeat segments:
python data_process_for_MIAT.py

(2) train and evaluation task DS1->DS2:
python MIAT_train_eval.py --run_id=0 --gpu=0 --epochs=150 --lr=0.001 --weight=0.005 --n=5 --lambda=1 --alpha=2 --prevat=1 --mix=1 --vat=1 --s=DS1 --t=DS2

(3) train and evaluation task MITDB->SVDB:
python MIAT_train_eval.py --run_id=0 --gpu=0 --epochs=150 --lr=0.001 --weight=0.005 --n=1 --lambda=5 --alpha=1 --prevat=1 --mix=1 --vat=1 --s=mitdb --t=svdb

```

