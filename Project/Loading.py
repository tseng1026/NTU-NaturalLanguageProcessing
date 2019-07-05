import numpy as np
import pandas as pd
from tqdm import tqdm

def LoadData(filename, taskname):
	load = pd.read_csv(filename, sep='\t', header=0)

	if taskname == "taskA":
		data = load[["tweet"]].tweet.tolist()
		task = load[["subtask_a"]].values.tolist()
	
	if taskname == "taskB":
		data = load.query("subtask_a == 'OFF'")[["tweet"]].tweet.tolist()
		task = load.query("subtask_a == 'OFF'")[["subtask_b"]].values.tolist()
	
	if taskname == "taskC":
		data = load.query("subtask_a == 'OFF' and subtask_b == 'TIN'")[["tweet"]].tweet.tolist()
		task = load.query("subtask_a == 'OFF' and subtask_b == 'TIN'")[["subtask_c"]].values.tolist()
	
	return data, task
