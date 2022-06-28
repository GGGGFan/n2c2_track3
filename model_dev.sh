for EPOCHS in 3 4  ; do
	for LR in 1e-5 2e-5 4e-5; do
	  	python model_dev.py \
		  --epoch=$EPOCHS \
		  --learning_rate=$LR
	done
done
