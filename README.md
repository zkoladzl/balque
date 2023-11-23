# BALQUE: Batch Active Learning by Querying Unstable Examples with Calibrated Confidence
# setup
- python == 3.7.7
- numpy == 1.18.1
- torch == 0.8.1
# active training
**installation**
```bash
git clone git@github.com:zkoladzl/balque.git
```
**data selection**
```bash
cd balque
python resnet18_training.py
```
>The setting of parameters, dataset, active learning methods,etc. is configured in `config.py`
