# Reproduce the CSLR experiments

The code is based on the Sign Language Transformers (CVPR'20) github repository, <https://github.com/neccam/slt>.

1. Clone the original SLT's github repository. 
```bash
git clone https://github.com/neccam/slt
cd slt
```

2. Download dataset.
```bash
mkdir data/PHOENIX2014T
cd data/PHOENIX2014T
bash ../download.sh
cd ../../
```

3. Set up the python virtual environment and install the required packages.
```bash
python3 -m venv venv
source venv/bin/activate
pip install pip --upgrade
pip install -r ../requirements.txt
```

* [Optional] Now you can first test if the original SLT code can work by running
```bash
python -m signjoey train configs/sign.yaml
```
If you encounter any problems, please refer to https://github.com/neccam/slt to resolve.

4. Copy the modified code to the folder.
```bash
cp ../run.sh .
cp -r ../configs .
cp -r ../signjoey .
```

5. Run the script.
```bash
bash run.sh
```
The script will loop the mask ratio variable, $r, from 0.0 to 0.9, and run the training and evaluation using both CTC and WCTC.
The results will be recorded in workdir/ctc/$r/train.log and workdir/wctc/$r/train.log, correspondingly.

