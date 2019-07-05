import argparse

def Args():
	parser = argparse.ArgumentParser()
	parser.add_argument("-d", default="./data/olid-training-v1.0.tsv")
	parser.add_argument("-m", default="DT")
	parser.add_argument("-t", default="taskA")

	args = parser.parse_args()
	return args
