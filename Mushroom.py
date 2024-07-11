import torch
from torch.utils.data import Dataset
import pandas as pd
class Mushroom(Dataset):
	# constructor
	def __init__(self, csv_file):
		self.data = pd.read_csv(csv_file)
		# get max values for each feature
		self.feature_maxes = self.data[["cap-diameter", "cap-shape", "gill-attachment","gill-color", "stem-height", "stem-width","stem-color", "season"]].max()
	# length of the dataset
	def __len__(self):
		return len(self.data)
	# iterate over dataset
	def __getitem__(self, idx):
		sample = self.data.iloc[idx]
		features = sample[["cap-diameter", "cap-shape", "gill-attachment","gill-color", "stem-height", "stem-width","stem-color", "season"]]
		# normalize features using maximum values
		features = features / self.feature_maxes
		# convert features to torch tensor
		features = torch.tensor(features.values).float()
		label = sample["class"]
		return features, label
