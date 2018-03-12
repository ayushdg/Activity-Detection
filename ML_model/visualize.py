from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os 

def readData(label,feature):
	suffix = '.csv'
	filename = os.path.join('Data/csv/',label,feature+suffix)
	dataframe = pd.read_csv(filename,header=None)
	dataframe.columns = ['id','x','y','z','a','label']
	return dataframe
	
def graph(*dataframe):
	for df in dataframe:
		plt.plot(df['acceleration'],label=df['label'][0],linewidth=0.75)

	plt.title('Accelerometer Data')
	plt.ylabel('Amplitude')
	plt.xlabel('Time')
	plt.legend(loc='best')
	plt.show()

def normalize_df(*dataframe):
	for df in dataframe:
		for column in ['x','y','z']:
			df[column] = (df[column]-df[column].mean())/df[column].std()

def find_acceleration(*dataframe):
	for df in dataframe:
		x = np.array(df['x'])
		y = np.array(df['y'])
		z = np.array(df['z'])
		df['acceleration'] = np.sqrt(np.multiply(x,x)+np.multiply(y,y)+np.multiply(z,z))

sitting = readData('sitting','accelerometer')
walking = readData('walking','accelerometer')
normalize_df(walking,sitting)
find_acceleration(walking,sitting)
graph(walking[:4000],sitting[:4000])
