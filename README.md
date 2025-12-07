This project implements and evaluates a BiLSTM sentiment classifier trained on multiple text corpora (GoEmotions, UCI Sentiment, and social media datasets). 
It investigates how dataset characteristics influence model behavior when predicting sentiment in music lyrics.

##################### How to train a model ##########################

All training is done via training.py.

Run this:

'python training.py --csv <PATH_TO_CSV> --out_prefix <NAME>'

example usage: python training.py --csv data/_processed_datasets/goemotions_processed_weighted.csv --out_prefix sentiment_weighted

This will save:

'model_<NAME>.pth'
'vocab_<NAME>.pkl'

in the project root.

Commands for all 5 models used in main.py

Reddit (GoEmotions – weighted)
python training.py --csv data/_processed_datasets/goemotions_processed_weighted.csv --out_prefix sentiment_weighted

Reddit (GoEmotions – unweighted)
python training.py --csv data/_processed_datasets/goemotions_processed_unweighted.csv --out_prefix sentiment_unweighted

Product Reviews (UCI Sentiment Dataset)
python training.py --csv data/_processed_datasets/uci_sentiment.csv --out_prefix sentiment_uci_sentiment

Social Media Comments (Sentiment Dataset)
python training.py --csv data/_processed_datasets/sentimentdataset_binary.csv --out_prefix sdsocial

Social Media Comments (Mental Health Dataset)
python training.py --csv data/_processed_datasets/mentalhealth_binary.csv --out_prefix mentalhealth

After each command finishes it will create corresponding model_*.pth and vocab_*.pkl files in the repo root:

model_sentiment_weighted.pth
vocab_sentiment_weighted.pkl

model_sentiment_unweighted.pth
vocab_sentiment_unweighted.pkl

model_sentiment_uci_sentiment.pth
vocab_sentiment_uci_sentiment.pkl

model_sdsocial.pth
vocab_sdsocial.pkl

model_mentalhealth.pth
vocab_mentalhealth.pkl

These are the filenames main.py expects.

##################### How to preprocess datasets ##########################

'python -m preprocessing.<FILENAME>'

example: 

python -m preprocessing.preproc_mentalhealth

############################# How to run project #######################

use 

python main.py

To view results of trained model (sentiment scores) corresponding to the text input.


