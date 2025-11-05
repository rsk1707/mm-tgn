'''
File to preprocess data for training and evaluation. Each dataset needs to be separately
processed and saved in a standard format for the model to consume. This common format will 
follow TGN's data format: Every dataset will be a dataset of interactions, where
column u is for source node ids, column i is for destination node ids, column ts is for timestamps,
and column idx is for edge indices.
'''