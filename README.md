# W-CTC: a Connectionist Temporal Classification Loss with Wild Cards

This repository contains the PyTorch implementation of W-CTC and the essential codes to reproduce the paper's experiments.
- [x] W-CTC PyTorch Implementation
- [x] ASR Experiments
- [ ] PR Experiments (release later)
- [x] OCR Experiments
- [x] CSLR Experiments

## Credits
The code is based on following repositories:
* CTC implementation    	: <https://github.com/vadimkantorov/ctc> 				
* ASR and PR experiments  : <https://github.com/huggingface/transformers/tree/master/examples/research_projects/wav2vec2> 
* OCR experiment          : <https://github.com/Media-Smart/vedastr>     
* CSLR experiment         : <https://github.com/neccam/slt> 					

## The repo structure
* wctc.py : The standalone implementation of W-CTC. Note that only forward algorithm is implemented, the backward is done via PyTorch's autograd mechanism. 
* ASR/: The code to reproduce ASR experiments.
* CSLR/: The code to reproduce CSLR expriments.

## Minimum effort to run (the ASR experiments)
```bash
cd ASR/
pip install -r requirements.txt
bash script.sh 0.2
```
Here 0.2 is a hyper-parameter that controls the ratio of masking on the labels.
It takes a value between 0 and 1.
When it's 0.0, no masking (label is clean).
When it's 1.0, the whole label is masked.

## Reproduce the experiments.
* For ASR, you can use above commands, or refer to the details in ASR/README.md.
* For OCR, please follow the instructions in OCR/README.md.
* For CSLR, please follow the instructions in CSLR/README.md.
