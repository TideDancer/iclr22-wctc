# Reproduce the OCR experiments

The code is based on the Vedastr (2020) github repository, <https://github.com/Media-Smart/vedastr>.

## Run

1. Clone the original vedastr's github repository. 
```bash
git clone https://github.com/Media-Smart/vedastr
cd vedastr
vedastr_root=${PWD}
```

2. Create a python virtual environment and install the required packages.
```bash
python3 -m venv venv
source venv/bin/activate
pip install pip --upgrade
pip install -r ../requirements.txt
```

3. Prepare dataset. Please follow instructions in the "Prepare data" section at <https://github.com/Media-Smart/vedastr>.
We list the key steps here:

* Download the lmdb data from <https://github.com/clovaai/deep-text-recognition-benchmark>.
(The download link should be <https://www.dropbox.com/sh/i39abvnefllx2si/AAAbAYRvxzRp3cIE5HzqUw3ra?dl=0>.)
* Download the ST dataset from <https://github.com/ayumiymk/aster.pytorch#data-preparation>. Please refer to their page for password.
* Put the downloaded lmdb data in a folder structure as shown in the "Prepare data" section at <https://github.com/Media-Smart/vedastr>.

In the end, you should have a folder "data/data_lmdb_release/" that contains training, validation, evaluation subfolders.

4. Copy the modified code to the folder.
```bash
cp ../run.sh .
cp -r ../configs .
cp -r ../veda/* vedastr/
cp -r ../tools .
```

5. Run the script to train.
```bash
bash run.sh wctc
```
The script will set the mask ratio variable, $r, to be 0.3, and run the training using WCTC.
The training results and checkpoints, will be saved in workdir/resnet_wctc/$r/.

The mask ratio $r can take values from 0 and 1.

* To run using standard CTC instead of WCTC, replace the "wctc" arg with "ctc".
* The training (150k steps) will take a few days to finish.
* Please refer to configs/resnet_wctc.py and confis/resnet_ctc.py for configuration details.

6. Run evaluation
After the training finished (make sure you have best_acc.pth in workdir/resnet_wctc/$r/), you can run:
```bash
bash run.sh test_wctc
```
to conduct the evaluations on all the test set.
The test result will be saved in workdir/resnet_wctc/$r/test_results.txt.

## Note
The random masking on the labels are NOT integrated into the data loading procedures, but at the training time.

To ensure the label is masked in the same way across different epochs,
the code relies on setting the random seed during training.
Specifically, the shuffling of training data is truned off, and random seed is set before every epoch.
Therefore, the masked part will be the same across epochs.



