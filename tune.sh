for EPOCHS in 3 4  ; do
	for LR in 1e-5 2e-5 4e-5; do
	  	python model_dev.py \
		  --epoch=$EPOCHS \
		  --learning_rate=$LR \
		  #--path_train_df=/data/N2C2-Track3-May3/train.csv \# modify this to set path of training dataframe
		  #--path_dev_df=/data/N2C2-Track3-May3/dev.csv \# modify this to set path of dev dataframe
		  #--path_train_add=/handcrafted_features/added_fts_tr.npy \# modify this to set path of handcrafted features
		  #--path_dev_add=/handcrafted_features/added_fts_dev.npy \# modify this to set path of handcrafted features
		  # if you choose clinical-bert
		  #--pretrained_model=emilyalsentzer/Bio_ClinicalBERT \
		  #--local_model=True \
		  # if you choose clinical-entitybert
		  #--pretrained_model=/downloaded_models/PubmedBERTbase-MimicBig-EntityBERT # modify this to set path of the downloaded entity bert
	done
done
