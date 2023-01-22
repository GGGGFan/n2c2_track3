python run_model_1.py \
  --path_train_add='/ext_info/added_fts_tr.npy'\
  --path_dev_add='/ext_info/added_fts_dev.npy'\
  --path_test_add='/ext_info/added_fts_test.npy'\
  --path_train_metamap='/ext_info/ner_metamap_train_v2.pickle'\
  --path_dev_metamap='/ext_info/ner_metamap_dev_v2.pickle'\
  --path_test_metamap='/ext_info/ner_metamap_test.pickle'
python run_model_2.py \
  --path_train_add='/ext_info/added_fts_tr.npy'\
  --path_dev_add='/ext_info/added_fts_dev.npy'\
  --path_test_add='/ext_info/added_fts_test.npy'\
  --path_train_metamap='/ext_info/ner_metamap_train_v2.pickle'\
  --path_dev_metamap='/ext_info/ner_metamap_dev_v2.pickle'\
  --path_test_metamap='/ext_info/ner_metamap_test.pickle'
python run_model_3.py  \
  --path_train_add='/ext_info/added_fts_tr.npy'\
  --path_dev_add='/ext_info/added_fts_dev.npy'\
  --path_test_add='/ext_info/added_fts_test.npy'\
  --path_train_metamap='/ext_info/ner_metamap_train_v2.pickle'\
  --path_dev_metamap='/ext_info/ner_metamap_dev_v2.pickle'\
  --path_test_metamap='/ext_info/ner_metamap_test.pickle'
python run_model_4.py  \
  --path_train_add='/ext_info/added_fts_tr.npy'\
  --path_dev_add='/ext_info/added_fts_dev.npy'\
  --path_test_add='/ext_info/added_fts_test.npy'\
  --path_train_metamap='/ext_info/ner_metamap_train_v2.pickle'\
  --path_dev_metamap='/ext_info/ner_metamap_dev_v2.pickle'\
  --path_test_metamap='/ext_info/ner_metamap_test.pickle'
python run_model_5.py  \
  --path_train_add='/ext_info/added_fts_tr.npy'\
  --path_dev_add='/ext_info/added_fts_dev.npy'\
  --path_test_add='/ext_info/added_fts_test.npy'\
  --path_train_metamap='/ext_info/ner_metamap_train_v2.pickle'\
  --path_dev_metamap='/ext_info/ner_metamap_dev_v2.pickle'\
  --path_test_metamap='/ext_info/ner_metamap_test.pickle'
python run_model_6.py  \
  --path_train_add='/ext_info/added_fts_tr.npy'\
  --path_dev_add='/ext_info/added_fts_dev.npy'\
  --path_test_add='/ext_info/added_fts_test.npy'\
  --path_train_metamap='/ext_info/ner_metamap_train_v2.pickle'\
  --path_dev_metamap='/ext_info/ner_metamap_dev_v2.pickle'\
  --path_test_metamap='/ext_info/ner_metamap_test.pickle'
python ensemble.py
