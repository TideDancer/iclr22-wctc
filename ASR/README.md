# Reproduce the ASR experiments

The code is based on the Huggingface repository at <https://github.com/huggingface/transformers/tree/master/examples/research_projects/wav2vec2>.

You need to have libsndfile installed in the system:
```bash
apt-get install libsndfile-dev
```
If encountering any error, please refer to libsndfile page <https://github.com/libsndfile/libsndfile> for details.

To run:
```bash
pip install -r requirements.txt
for r in {0..9}; do bash script.sh 0.$r; done
```
Here "0.$r" controls the ratio of masking on the labels.
It takes a value between 0 and 1.
When it's 0.0, no masking (label is clean).
When it's 1.0, the whole label is masked.

The script loop "0.$r" from 0.0 to 0.9, and reproduce the ASR experiment using WCTC.

##
If you want to use the standard CTC instead of WCTC, change the line 6 in script.sh:
```
export WCTC=True
```
to
```
export WCTC=False
```

## Settings to reproduce ASR experiments
We use learning rate = 1e-4, effective batch size = 32, train 50 epochs.
Other settings can be found in script.sh and the paper appendix.


