# N2C2 Track3

This model combine the output of the pretrained [ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) or [EntityBERT](https://physionet.org/content/entity-bert/1.0.1/) and 3 handcrafted features (percentile rank of each sample in corresponding notes, indicators that whether there is any overlapped concepts (recognized by [MetaMap](https://lhncbc.nlm.nih.gov/ii/tools/MetaMap.html)) in terms of [ICD-10 blocks and ICD-10 chapters](https://crvsgateway.info/Volume-1-Tabular-list~584)). You may need to download the [EntityBERT](https://physionet.org/content/entity-bert/1.0.1/) and load the pre-trained model locally as it has not been uploaded to the Huggingface Hub.

`tune.sh` is used to tune the model. The training data is splitted to train / dev set at 4:1 ratio by hadm_id and the dev set is used as the testing set. To tune the models, please run:
```
chmod +x tune.sh
sh tune.sh
```
Currently, the best preformance on the dev set comes from the fine-tuned EntityBERT model with a learning rate of 2e-5 and an training epoch of 3. Run below lines to train on the whole training set and report the micro-F1 on the dev set. The micro-F1 should be around 0.782.
```
chmod +x validate.sh
sh validate.sh
```
The best F1 from pretrained ClinicalBERT is around 0.76-0.77. We also tried the [Clinical-Longformer](https://huggingface.co/yikuan8/Clinical-Longformer) but its perofmance is no better than 0.70.
