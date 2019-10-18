import numpy as np
import pandas as pd

ls = []
lsy = []

with open('hw1_ridge_x.dat') as f:
	for i in f:
		ls.append(i.strip('\n').strip(' '))
	ls.remove('')

with open('hw1_ridge_y.dat') as g:
	for i in g:
		lsy.append(i.strip('\n').strip(' '))
	lsy.remove('')

