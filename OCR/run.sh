for MR in 0.3 #0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do

# train
if [ "$1" = "ctc" ]; then
	python3 tools/train.py configs/resnet_$1.py "0" --mask_ratio $MR &
fi

if [ "$1" = "wctc" ]; then
	python3 tools/train.py configs/resnet_$1.py "0" --mask_ratio $MR &
fi

# test
if [ "$1" = "test_ctc" ]; then
	python3 tools/test.py configs/resnet_ctc.py workdir/resnet_ctc/$MR/best_acc.pth 1 > workdir/resnet_ctc/$MR/test_results.txt &
fi

if [ "$1" = "test_wctc" ]; then
	python3 tools/test.py configs/resnet_wctc.py workdir/resnet_wctc/$MR/best_acc.pth 1 > workdir/resnet_wctc/$MR/test_results.txt &
fi

done

