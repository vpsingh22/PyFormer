# PyFormer

PyFormer is a machine translation model that converts english sentences to python. 

## Dataset

The dataset is a small set of english text and their respective python code, approx 4000 entries. You can download the dataset [here]()

## Data Cleaning and Preparation
The data was cleaned manually by removing unneccessary indentation and comments. 

The data needs to be tokenized  properly before feeding it to the model. The english sentences were tokenized using spacy.
To tokenize the python code, a lexical analyzer tool is written which tokenizes the code according to python. If you wish to use the model for your language you can modify the tool [here](https://github.com/vpsingh22/PyFormer/blob/master/data/lexical_analyzer.py)

## Embeddings Training
The embeddings are trained using glove. If you wish not to train the embeddings you can set embeddings_training to False in config.
Glove uses weighted mean square error to find the corelation between the tokens for the vocabulary. The trained embeddings weights are directly copied to the Decoder embeddings layer of the model.

## Model Architecture

## Results
The model gives pretty good results.<br>
![ss14](./sample_outputs/ss14.png)<br>
The above is a sample example. You can view the results of 35+ examples [here](https://github.com/vpsingh22/PyFormer/tree/master/sample_outputs)


