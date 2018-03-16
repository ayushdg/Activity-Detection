#from azureml.datacollector import ModelDataCollector
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from numpy import convolve
from scipy.interpolate import spline
import itertools
import math
from statistics import mode
from scipy.fftpack import *
import scipy
import os
import csv
import pickle
from sklearn.externals import joblib
from scipy.signal import butter, lfilter, freqz

# Functions
def stft(x, fftsize=256, overlap=2):   
    hop = fftsize // overlap
    w = scipy.hanning(fftsize+1)[:-1]
    return np.array([np.fft.fft(w*x[i:i+fftsize]) for i in range(0, len(x)-fftsize, hop)])

def get_fft(x):   
    return np.fft.fft(x)
    
def get_frequency(time):
    total_time = time[-1] - time[0]
    total_samples = len(time)
    frequency = (total_samples * 1000) // total_time
    return total_time, total_samples, frequency

def basicFeats(x, fftsize=256, overlap=2):
    meanamp = []
    maxamp = []
    minamp = []
    stdamp = []
    mad=[]
    hop = fftsize // overlap
    for i in range(0, len(x)-fftsize, hop):
        meanamp.append(np.array(np.mean(x[i:i+fftsize])))
        maxamp.append(np.array(np.max(x[i:i+fftsize])))
        minamp.append(np.array(np.min(x[i:i+fftsize])))
        stdamp.append(np.array(np.std(x[i:i+fftsize])))
        mad.append(np.array(np.median(np.abs(x[i:i+fftsize] - np.median(x[i:i+fftsize])))))
    return meanamp, maxamp, minamp, stdamp, mad

def feature(x, y, z, m, fftsize=256, overlap=2):
    energyamp_m = []
    kurtosisamp_m = []
    skewamp_m = []
    meanamp_x, maxamp_x, minamp_x, stdamp_x, mad_x = basicFeats(x, fftsize)
    meanamp_y, maxamp_y, minamp_y, stdamp_y, mad_y = basicFeats(y, fftsize)
    meanamp_z, maxamp_z, minamp_z, stdamp_z, mad_z = basicFeats(z, fftsize)
    meanamp_m, maxamp_m, minamp_m, stdamp_m, mad_m = basicFeats(m, fftsize)
    
    hop = fftsize // overlap
    for i in range(0, len(m)-fftsize, hop):
        energyamp_m.append(np.array(np.sum(np.power(m[i:i+fftsize],2))))
        kurtosisamp_m.append(kurtosis(m[i:i+fftsize]))
        skewamp_m.append(skew(m[i:i+fftsize]))
                
    return [meanamp_x, maxamp_x, minamp_x, stdamp_x, mad_x, meanamp_y, maxamp_y, minamp_y, stdamp_y, mad_y, 
            meanamp_z, maxamp_z, minamp_z, stdamp_z, mad_z, meanamp_m, maxamp_m, minamp_m, stdamp_m, mad_m, 
            energyamp_m, kurtosisamp_m, skewamp_m]

def normalize_fft_feat(features_list):
    ret_feat = []
    for feat in features_list:
        ret_feat.append(feat / np.linalg.norm(feat))
    
    return ret_feat
    
def normalize_feat(features_list):
    ret_feat = []
    for feat in features_list:
        ret_feat.append(feat / np.linalg.norm(feat))
    
    return ret_feat
            
def five_point_smoothing(m):
    m_smooth = np.zeros(len(m))
    m_smooth = m;
    for i ,val in enumerate(m_smooth[2 : (len(m_smooth) -2)]):
        m_smooth[i] = (m_smooth[i-2] + m_smooth[i-1] + m_smooth[i] + m_smooth[i+1] + m_smooth[i+2])/5
        
    return m_smooth

def inter_quartile_range(x, y, z, fftsize=256, overlap=2):
    """Calculates inter-quartile range"""
    iqr_x = []
    iqr_y = []
    iqr_z = []
    hop = fftsize // overlap
    for i in range(0, len(x)-fftsize, hop):
        iqr_x.append(np.subtract(*np.percentile(x[i:i+fftsize], [75, 25])))
        iqr_y.append(np.subtract(*np.percentile(y[i:i+fftsize], [75, 25])))
        iqr_z.append(np.subtract(*np.percentile(z[i:i+fftsize], [75, 25])))
    return iqr_x, iqr_y, iqr_z

def sma(x, y, z, fftsize=256, overlap=2):
    """Calculates signal magnitude area"""
    abs_x = []
    abs_y = []
    abs_z = []
    hop = fftsize // overlap
    for i in range(0, len(x)-fftsize, hop):
        abs_x.append(np.absolute(x[i:i+fftsize]))
        abs_y.append(np.absolute(y[i:i+fftsize]))
        abs_z.append(np.absolute(z[i:i+fftsize]))
    return [np.mean(x+y+z) for x,y,z in zip(abs_x, abs_y, abs_z)]

def calc_jerk(acc, ts):
    jk = [0]* len(acc)
    for i in range(1,len(acc)):
        jk[i-1] = 1000*(acc[i] - acc[i-1])/(ts[i] - ts[i-1])
    return jk

def get_jerk(x,fftsize=256, overlap=2):
    jerk_x=[]
    hop = fftsize // overlap
    for i in range(0, len(x)-fftsize, hop):
        jerk_x.append(np.array(np.mean(x[i:i+fftsize])))   
    
    return jerk_x

def get_window_label(labels, fftsize=256, overlap=2):
    hop = fftsize // overlap
    ret_labels = []
    for i in range(0, len(labels)-fftsize, hop):
        ret_labels.append(mode(labels[i:i+fftsize]))
    
    return ret_labels


#global dict
label = {'sitting' : 0, 'walking' : 1, 'laying_down' : 2, 'standing' : 3, 'unknown' : 4}


def compute_feats(*dataframes, show_plots=False):
    show_plots=False
    global label
    #df = pd.DataFrame(columns=list(range(60)))
    for counter, data in enumerate(dataframes):
        
        time, x, y, z, _, str_label = np.array(data[0]), np.array(data[1]), np.array(data[2]), np.array(data[3]), \
                                        np.array(data[4]), np.array(data[5])        
        
        total_time, total_sample, freq = get_frequency(time)
        #5 seconds time frame
        window_size = 5 * freq
        
        labels = [label[str_label[i]] for i in range(len(str_label))]
        window_labels = get_window_label(labels, fftsize=window_size)
        #Calculate the Frequency domain
        x = five_point_smoothing(x)
        y = five_point_smoothing(y)
        z = five_point_smoothing(z)
        
        ft_x = get_fft(x)
        ft_x = np.abs(ft_x)
        ft_y = get_fft(y)
        ft_y = np.abs(ft_y)
        ft_z = get_fft(z)
        ft_z = np.abs(ft_z)
        
        
        #Amplitude Calculation
        mpre = x * x + y * y + z * z
        m = np.sqrt(mpre)
        #Low pass filtering (5 point smoothing)
        if show_plots:
            plt.plot(m[:4000])
            plt.title('Accelerometer Data - '+str_label[0])
            plt.ylabel('Amplitude')
            plt.xlabel('Time')
            plt.show()
        
        #Jerk
        jerk_x = calc_jerk(x, time)
        jerk_y = calc_jerk(y, time)
        jerk_z = calc_jerk(z, time)
        jerk_m = np.array(jerk_x) * np.array(jerk_x) + np.array(jerk_y) * np.array(jerk_y) + np.array(jerk_z) * np.array(jerk_z)
        
        #FFT features
        stft_signal = stft(m, fftsize=window_size)
        energy_signal = []
        for i, amp in enumerate(stft_signal):
            energy_signal.append(np.sum(np.power(abs(stft_signal[i]),2)))
        
        features_list = feature(x, y, z, m, fftsize=window_size)
        features_list.append(energy_signal)
        diff_x = np.subtract(features_list[1], features_list[2])
        diff_y = np.subtract(features_list[6], features_list[7])
        diff_z = np.subtract(features_list[11], features_list[12])
        diff_m = np.subtract(features_list[16], features_list[17])
        features_list.append(diff_x)
        features_list.append(diff_y)
        features_list.append(diff_z)
        features_list.append(diff_m)
        
        iqr_x, iqr_y, iqr_z = inter_quartile_range(x, y, z, fftsize=window_size)
        smaq = sma(x, y, z, fftsize=window_size)
        features_list.append(iqr_x)
        features_list.append(iqr_y)
        features_list.append(iqr_z)
        features_list.append(smaq)
        
        #Add Jerk features to the list
        jerk_features_list = feature(jerk_x, jerk_y, jerk_z, jerk_m, fftsize=window_size)

        # Normalize other features
        norm_features_list = normalize_feat(features_list)
        norm_jerk_features_list = normalize_feat(jerk_features_list)
        
        ########################################################################
        # Frequency Domain
        ########################################################################
        
        #Frequency amplitude
        mpre = ft_x * ft_x + ft_y * ft_y + ft_z * ft_z
        m_f = np.sqrt(mpre)
        #Low pass filter (5 point Smoothing)
        
        if show_plots:
            plt.plot(m_f[:4000])
            plt.title('Accelerometer Data - '+str_label[0])
            plt.ylabel('Freq Amplitude')
            plt.xlabel('Time')
            plt.show()
        
        fft_features_list = feature(ft_x, ft_y, ft_z, m_f, fftsize=window_size)
        
        fft_diff_x = np.subtract(features_list[1], features_list[2])
        fft_diff_y = np.subtract(features_list[6], features_list[7])
        fft_diff_z = np.subtract(features_list[11], features_list[12])
        fft_diff_m = np.subtract(features_list[16], features_list[17])
        fft_features_list.append(fft_diff_x)
        fft_features_list.append(fft_diff_y)
        fft_features_list.append(fft_diff_z)
        fft_features_list.append(fft_diff_m)
        
        fft_iqr_x, fft_iqr_y, fft_iqr_z = inter_quartile_range(ft_x, ft_y, ft_z, fftsize=window_size)
        fft_smaq = sma(ft_x, ft_y, ft_z, fftsize=window_size)
        fft_features_list.append(fft_iqr_x)
        fft_features_list.append(fft_iqr_y)
        fft_features_list.append(fft_iqr_z)
        fft_features_list.append(fft_smaq)
        
        #Jerk
        win_jerk_x = get_jerk(jerk_x, fftsize=window_size)
        win_jerk_y = get_jerk(jerk_y, fftsize=window_size)
        win_jerk_z = get_jerk(jerk_z, fftsize=window_size)
        
        fft_jerk_x = get_fft(win_jerk_x)
        fft_jerk_x = np.abs(fft_jerk_x)        
        fft_jerk_y = get_fft(win_jerk_y)
        fft_jerk_y = np.abs(fft_jerk_y)
        fft_jerk_z = get_fft(win_jerk_z)
        fft_jerk_z = np.abs(fft_jerk_z)
        #Add Jerk features to the list
        fft_features_list.append(fft_jerk_x)
        fft_features_list.append(fft_jerk_y)
        fft_features_list.append(fft_jerk_z)
        jerk_iqr_x, jerk_iqr_y, jerk_iqr_z = inter_quartile_range(jerk_x, jerk_y, jerk_z, fftsize=window_size)
        jerk_smaq = sma(jerk_x, jerk_y, jerk_z, fftsize=window_size)
        
        #Normalize fft features
        fft_norm_features_list = normalize_fft_feat(fft_features_list)
        
        if show_plots:
            plt.plot(energy_signal[:4000])
            plt.xlabel('Freq')
            plt.ylabel('Energy')
            plt.title('Frequency vs Energy - '+str_label[0])
            plt.show()
        
        
        ####################################################################################
        norm_features_list.extend(fft_norm_features_list)
        norm_features_list.extend(norm_jerk_features_list)
        #Take transpose
        norm_features_list = list(map(list, zip(*norm_features_list)))
        
        #Put in a dataframe
        temp_df = pd.DataFrame(norm_features_list)
        temp_df[len(temp_df.columns)] = window_labels

        if counter == 0:
            df = temp_df
        else:
            df = df.append(temp_df)
            
    return df

def test_model_api(files):
    ''' This function is for API call made by Azure '''

    string_annot = ['acc_', 'gyr_', 'mag_']
    dfs = []
    for i, file in enumerate(files):
        print(file.head())
        file = compute_feats(file, show_plots=False)
        file.columns = [string_annot[i] + str(col) for col in file.columns]
        dfs.append(file)
    #print(dfs[0].head())
    min_len = min([len(df) for df in dfs])
    final_df = dfs[0]
    for df in dfs[1:]:
        final_df = pd.concat([final_df.loc[:min_len-1, :final_df.columns[-2]],
                              df.loc[:min_len-1, :]], axis=1)

    final_df.columns = [i if i == len(final_df.columns)-1 else final_df.columns[i] for i in range(len(final_df.columns))]
    label_col = len(final_df.columns) - 1
    labels = final_df[label_col]
    labels = labels.astype('int')
    data = final_df.drop(label_col, axis=1)
    return data



def init():
    from sklearn.externals import joblib
    import pickle
    global model, inputs_dc, prediction_dc
    model = joblib.load('testModel2.pkl')

    #inputs_dc = ModelDataCollector('testModel.pkl',identifier="inputs")
    #prediction_dc = ModelDataCollector('testModel.pkl', identifier="prediction")

def load_data():
    data_dict = {}
    acc = pd.read_csv('data/csv/ashish/accelerometer-lay.csv', nrows=5000)
    gyro = pd.read_csv('data/csv/ashish/gyroscope-lay.csv', nrows=5000)
    mag = pd.read_csv('data/csv/ashish/magnetic-lay.csv', nrows=5000)
    data_dict['acceleration'] = acc.values.tolist()
    data_dict['gyroscope'] = gyro.values.tolist()
    data_dict['magnetic'] = mag.values.tolist()
    return data_dict

def run(input_array):
    from sklearn.externals import joblib

    global clf2, inputs_dc, prediction_dc
    try:
        import json
        int_to_label = {0 : 'sitting', 1 : 'walking', 2 : 'laying_down', 3 : 'standing', 4 : 'unknown'}
        #input_array = json.loads(input_array)
        acc = input_array['acceleration']
        gyro = input_array['gyroscope']
        mag = input_array['magnetic']
        model = joblib.load('azure_models1/acc_RFClassifier-onevsone.pkl')
        df_acc = pd.DataFrame(acc)
        df_gyro = pd.DataFrame(gyro)
        df_mag = pd.DataFrame(mag)
        _, _, acc_freq = get_frequency(np.array(df_acc[0]))
        #_, _, gyro_freq = get_frequency(np.array(df_gyro[0]))
        #_, _, mag_freq = get_frequency(np.array(df_mag[0]))
        
        #if(len(df_acc)< 5 * acc_freq or len(df_gyro) < 5 * gyro_freq or len(df_mag) < 5 * mag_freq):
        #    return 'Unknown'
        if(len(df_acc)< 5 * acc_freq):
            return 'Unknown'
        #test_df = test_model_api([df_acc, df_gyro, df_mag])
        test_df = test_model_api([df_acc])
        pred_labels = model.predict(test_df)
        print(pred_labels)
        return int_to_label[mode(pred_labels)]
        input_array = input_array['acceleration'][0][1]
        input_array = np.asarray(input_array)
        prediction = model.predict(input_array)
        inputs_dc.collect(input_array)
        prediction_dc.collect(prediction)
        return prediction.tolist()
    except Exception as e:
        return (str(e))

data = load_data()
print(run(data))
