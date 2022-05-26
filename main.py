'''
code for adv-nlp final project
Transformer-based model for molecular property prediction 
'''
#!git clone https://github.com/jessevig/bertviz bertviz_repo

from rdkit import Chem
import sys
from apex import amp
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline, RobertaModel, RobertaTokenizer
from bertviz import head_view
import numpy as np
import pandas as pd
from typing import List
import deepchem as dc
import os
from deepchem.molnet import load_bbbp, load_clearance, load_clintox, load_delaney, load_hiv, load_qm7, load_tox21
import sklearn
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import roc_auc_score,average_precision_score 
from simpletransformers.classification import ClassificationModel
from molnet_dataloader import load_molnet_dataset

import logging


model = AutoModelForMaskedLM.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
tokenizer = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")

fill_mask = pipeline('fill-mask', model=model, tokenizer=tokenizer)


#!wget https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/vocab.txt

# change the name of the datasets accordingly
tasks, (train_df, valid_df, test_df), transformers = load_molnet_dataset("tox21", tasks_wanted=None)


featurizer = dc.feat.CircularFingerprint(size=1024)
train_ecfp = featurizer.featurize(train_df.text.values)
val_ecfp = featurizer.featurize(valid_df.text.values)
test_ecfp = featurizer.featurize(test_df.text.values)


# random forest baseline 
rf_model = RandomForestClassifier(n_estimators=200)
rf_model.fit(X= train_ecfp, 
             y = train_df.labels.values)

test_pred = rf_model.predict(test_ecfp)

rf_auc = roc_auc_score(y_true=test_df.labels.values, y_score=test_pred,average='weighted')
rf_ap = average_precision_score(y_true=test_df.labels.values, y_score=test_pred, average='weighted')
print(rf_auc, rf_ap)


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


model = ClassificationModel('roberta', 'seyonec/PubChem10M_SMILES_BPE_396_250', 
                            args={'evaluate_each_epoch': True, 
                                  'evaluate_during_training_verbose': True, 
                                  'no_save': True, 'num_train_epochs': 10,
                                  'auto_weights': True}) # You can set class weights by using the optional weight argument

print(model.tokenizer)

# check if our train and evaluation dataframes are setup properly. There should only be two columns for the SMILES string and its corresponding label.
print("Train Dataset: {}".format(train_df.shape))
print("Eval Dataset: {}".format(valid_df.shape))
print("TEST Dataset: {}".format(test_df.shape))

#!wandb login

# Create directory to store model weights 
#!mkdir BPE_PubChem_10M_tox21_run

# Train the model
model.train_model(train_df, eval_df=valid_df, 
                  output_dir='/content/BPE_PubChem_10M_TOX21_run', 
                  args={'wandb_project': 'yoyo_test1'})


# accuracy
result, model_outputs, wrong_predictions = model.eval_model(test_df, acc=sklearn.metrics.accuracy_score)

# ROC-PRC
result, model_outputs, wrong_predictions = model.eval_model(test_df, acc=sklearn.metrics.average_precision_score)

# Lets input a molecule with a toxicity value of 1
predictions, raw_outputs = model.predict(['C1=C(C(=O)NC(=O)N1)F'])

print(predictions)
print(raw_outputs)

