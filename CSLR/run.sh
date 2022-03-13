# run WCTC
for r in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
	python -m signjoey --gpu_id 0 --mask_ratio $r --wctc train configs/sign.yaml.wctc.$r
done

# run standard CTC
for r in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
	python -m signjoey --gpu_id 0 --mask_ratio $r train configs/sign.yaml.ctc.$r
done


