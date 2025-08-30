#!/usr/bin/env python
# coding: utf-8
"""
Audio Data Preprocessing Script
This script processes voice audio data for voice disorder classification tasks.
It handles two datasets: Chinese EENT dataset and German SVD dataset.
The preprocessing includes:
1. Reading audio files and extracting metadata
2. Segmenting audio by silence removal
3. Feature extraction (mel spectrograms and pitch-related features)
4. Generating pickle files for models training
"""
# In[1]:
# Import required libraries
from pyAudioAnalysis import audioBasicIO as aIO
from pyAudioAnalysis import audioSegmentation as aS
import scipy.io.wavfile as wavfile
import scipy
import numpy as np
import os
import soundfile as sf
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import traceback
import logging
logging.basicConfig(filename='D:/data/audio//logger.log', level=logging.DEBUG)
import seaborn as sns
import librosa
import matplotlib.pyplot as plt
import librosa.display
import math
import tensorflow as tf
import glob
import joblib
import parselmouth
from urllib.parse import quote
from parselmouth.praat import call
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# In[5]:

raw_wav_folder = r'D:/data/audio/final_EENT' # Path to Chinese dataset
processed_wav_folder = r'D:/data/audio/final_EENT_processed'

# In[6]:

disease_type = ['Functional Dysphonia','Glottal Incompetence','Nodules','Normal','Polyps','Sulcus', 'VFP']
label_dict = {'Normal': 0, 'Functional Dysphonia': 1, 'Glottal Incompetence':2, 'Nodules':3, 
              'Polyps': 4, 'Sulcus': 5, 'VFP' : 6}
voice_type = ['vowel-a','vowel-i','phrase','passage','chat','mpt']
feature_type = ['mel','pitch']#, 'pitch'

# In[8]:
def get_all_file_properties (raw_wav_folder, disease_type):
    """
    Traverse the original audio folder to collect information about each audio file,
    building a metadata table containing all original audio files.

    Parameters:
    raw_wav_folder (str): Path to the raw audio data folder
    disease_type (list): List of disease types to process

    Returns:
    pd.DataFrame: DataFrame containing file paths, patient IDs, disease types, voice types, durations, and sample rates
    """
    file_path = []
    p_ids = []
    vtype = []
    dtype = []
    sample_duration = []
    sample_sr = []

    for disease in disease_type:
        # Build disease type directory path
        disease_dir = os.path.join(raw_wav_folder, disease)
        # Get all patient ID subdirectories
        ids_path = glob.glob(os.path.join(disease_dir, '*'))

        for id_path in ids_path:
            # Normalize path and extract patient ID
            normalized_id_path = os.path.normpath(id_path.rstrip(os.sep))
            patient_id = os.path.basename(normalized_id_path)

            # Traverse files in patient directory
            for root, dirs, files in os.walk(id_path):
                for file in files:
                    if file.endswith(".wav"):
                        id_wav_path = os.path.join(root, file)
                        if 'post-tx' not in id_wav_path:
                            file_path.append(id_wav_path)
                            p_ids.append(patient_id)
                            vtype.append(os.path.splitext(file)[0])  # Extract voice type (filename without .wav)
                            dtype.append(disease)
                            # Read audio parameters
                            with sf.SoundFile(id_wav_path) as f:
                                sr = f.samplerate
                                flen = len(f) / sr
                                sample_duration.append(flen)
                                sample_sr.append(sr)

    df = pd.DataFrame(list(zip(file_path, p_ids, dtype, vtype, sample_duration, sample_sr)),
                      columns=['Reading path', 'ID', 'Disease', 'Voice type', 'Duration', 'Sample Rate'])

    return df

# Generate metadata for raw audio files
df_raw = get_all_file_properties(raw_wav_folder, disease_type)       # no post-tx 

save_path = "D:/data/audio/final_EENT_processed/eent_original_recording.csv"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
df_raw.to_csv(save_path, index=False, encoding='utf-8-sig')
print(f"Saved to：{os.path.abspath(save_path)}")

# In[9]:
df_raw.head()

# In[ ]:
def split_by_silence(df, target_voice_type, save_root_folder, smooth_window = 0.5, weight = 0.1, min_duration = 1.0, execution = None) :
    """
    Split audio into valid segments by silence detection.

    Parameters:
    df (pd.DataFrame): DataFrame containing audio file information
    target_voice_type (list): List of voice types to process
    save_root_folder (str): Root folder to save segmented audio files
    smooth_window (float): Smoothing window for silence detection
    weight (float): Weight parameter for silence detection
    min_duration (float): Minimum duration for valid segments
    execution (int): Flag to control whether to actually write files (None means skip file writing)

    Returns:
    pd.DataFrame: DataFrame containing segmented audio information
    """

    file_path = []
    p_ids = []
    v_type = []
    d_type = []
    sample_sr = []
    start = []
    end = []
    sample_duration = []
    
    for vtype in target_voice_type:
        # Create output folder for each voice type
        out_folder = os.path.join(save_root_folder, 'segmented', vtype)
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        
        for i in range(len(df)):
            if vtype in df['Voice type'][i]:
                try:
                    [fs, x] = aIO.read_audio_file(df['Reading path'][i])
                    # Perform silence removal
                    segments = aS.silence_removal(x, fs, 0.020, 0.020, smooth_window = smooth_window, weight = weight, plot = False)

                    for j, s in enumerate(segments):
                        # Only keep segments longer than minimum duration
                        if s[1]-s[0] >= min_duration: # can minimize empty output
                            out_filename = os.path.join(out_folder,'{0}_{1}_{2:.2f}-{3:.2f}.wav'.
                                              format(df['Disease'][i], df['ID'][i], s[0], s[1]))
                            if execution is not None: # in case files have been generated, but just need the dataframe
                                wavfile.write(out_filename, fs, x[int(fs * s[0]):int(fs * s[1])])
                            file_path.append(out_filename)
                            p_ids.append(df['ID'][i])
                            v_type.append(vtype)
                            d_type.append(df['Disease'][i])
                            sample_sr.append(fs)
                            start.append(s[0])
                            end.append(s[1])
                            sample_duration.append(s[1]-s[0])
                    
                except:
                    print(f'Something went wrong processing {df["Reading path"][i]}')

    # segmentation metadata
    df_gen = pd.DataFrame(list(zip(file_path, p_ids, d_type, v_type, start, end, sample_duration, sample_sr)), 
                           columns =['Reading path', 'ID', 'Disease', 'Voice type','Start', 'End', 'Duration','Sample Rate'])

    csv_save_dir = os.path.join(save_root_folder, 'metadata')
    os.makedirs(csv_save_dir, exist_ok=True)
    csv_filename = f'split_segments_{"_".join(target_voice_type)}.csv'
    csv_path = os.path.join(csv_save_dir, csv_filename)
    df_gen.to_csv(csv_path, index=False)
    print(f'DataFrame saved to: {csv_path}')

    return df_gen

# Perform silence-based segmentation on vowel audio files
df_split_vowel = split_by_silence(df_raw, voice_type[0:2], processed_wav_folder, execution = 1) 

# In[102]:

df_split_vowel.head()

# In[10]:
def read_split_vowel_to_df(base_folder):
    """
    For already processed data, read information from segmented files directly into DataFrame
    to avoid redundant processing.

    Parameters:
    base_folder (str): Base folder containing segmented audio files

    Returns:
    pd.DataFrame: DataFrame with segmented audio information
    """

    file_path = []
    p_ids = []
    v_type = []
    d_type = []
    sample_sr = []
    start = []
    end = []
    sample_duration = []
    
    for vtype in voice_type[0:2]:
        vowel_folder = os.path.join(base_folder, vtype)

        for roots, dirs, files in os.walk(vowel_folder):
            for file in files:
                if file.endswith(".wav"):
                    file_full_path = os.path.join(roots, file)
                    file_path.append(file_full_path)
                    p_ids.append(file.split('_')[1])
                    v_type.append(roots.split(os.sep)[-1])
                    d_type.append(file.split('_')[0])
                    f =  sf.SoundFile(file_full_path)
                    sr = f.samplerate
                    flen = len(f)/f.samplerate
                    sample_duration.append(flen)
                    sample_sr.append(sr)
                    start.append(file.split('_')[-1].split('-')[0])
                    end.append(file.split('_')[-1].split('-')[-1].split('.wav')[0])

    df_gen = pd.DataFrame(list(zip(file_path, p_ids, d_type, v_type, start, end, sample_duration, sample_sr)), 
                           columns =['Reading path', 'ID', 'Disease', 'Voice type','Start', 'End', 'Duration','Sample Rate'])

    metadata_dir = os.path.join(base_folder, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    csv_filename = "read_segmented.csv"
    csv_path = os.path.join(metadata_dir, csv_filename)
    df_gen.to_csv(csv_path, index=False)
    print(f"Saved to：{csv_path}")

    return df_gen

df_split_vowel = read_split_vowel_to_df('D:/data/audio/final_EENT_processed/segmented')

# In[11]:

df_split_vowel.head()

# In[12]:
def inspect_plot_duration(voice_type, df_v, df_s, ideal_q = 0.75,output_dir = None):
    """
    Check duration distribution for different voice types and calculate ideal segmentation lengths.

    Parameters:
    voice_type (list): List of voice types to analyze
    df_v (pd.DataFrame): DataFrame with vowel data
    df_s (pd.DataFrame): DataFrame with segmented data
    ideal_q (float): Quantile to use for ideal duration calculation
    output_dir (str): Directory to save plots and statistics

    Returns:
    pd.DataFrame: Summary of duration statistics
    """

    v_type = []
    ideal_dur = []

    for vtype in voice_type:
        v_type.append(vtype)
        print(f'-------{vtype}---------\n')

        # Select data based on voice type
        if vtype in ['vowel-a', 'vowel-i']:
            data = df_v[df_v['Voice type'] == vtype]['Duration']
        else:
            data = df_s[df_s['Voice type'].str.contains(vtype)]['Duration']

            data = df_s[df_s['Voice type'].str.contains(vtype)]['Duration']

        # Calculate and print quantiles
        quantiles = data.quantile([0.25, 0.5, 0.75])
        print(f"Quantile statistics:\n{quantiles}\n")

        # Plot distribution
        plt.figure()
        sns.distplot(data)
        plt.title(f'Duration Distribution - {vtype}')

        # Save distribution plot
        if output_dir is not None:
            plot_path = os.path.join(output_dir, f'{vtype}_distribution.png')
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved distribution plot to: {plot_path}")

        # Record ideal duration quantile
        ideal_dur.append(data.quantile(ideal_q))

    df_dur = pd.DataFrame({
        'Voice type': v_type,
        f'{ideal_q:.0%} duration': ideal_dur
    })

    if output_dir is not None:
        csv_path = os.path.join(output_dir, 'duration_summary.csv')
        df_dur.to_csv(csv_path, index=False)
        print(f"\nSaved duration statistics to: {csv_path}")

    return df_dur

# Analyze duration distributions
df_dur = inspect_plot_duration(voice_type, df_split_vowel, df_raw,
                               output_dir='D:/data/audio/final_EENT_processed/metadata')

# In[49]:
# German SVD dataset configuration
raw_wav_folder_ger = r'D:/data/audio/SVD'  # Path to German SVD dataset

disease_type_ger = ['Cordectomy', 'Dysarthrophonia', 'Dysodie', 'Dysphonia', 'Functional dysphonia', 'Healthy',
                    'Hyperfunctional dysphonia', 'Hypofunctional dysphonia', 
                    'Hypotonic dysphonia', 'Laryngitis', 'Polyp', 
                    'Reinke edema', 'Spasmodic dysphonia', 'Uncommon', 'VC carcinoma', 'VFP', ]

label_dict_ger = {'Healthy':0, 'Cordectomy':1, 'Dysarthrophonia':2, 'Dysodie':3, 'Dysphonia':4, 
                  'Functional dysphonia':5, 'Hyperfunctional dysphonia':6, 'Hypofunctional dysphonia':7,
                    'Hypotonic dysphonia':8, 'Laryngitis':9, 'Polyp':10, 
                    'Reinke edema':11, 'Spasmodic dysphonia':12, 'Uncommon':13, 'VC carcinoma':14, 'VFP':15}

voice_type_ger = ['a','i','iau','phrase','u']
tone_type_ger = ['h', 'l', 'lhl', 'n']

# In[61]:
def get_all_file_properties_ger (raw_wav_folder, disease_type, voice_type, tone_type):
    """
    Processing function for German dataset, adapted for different folder structure and disease classification.

    Parameters:
    raw_wav_folder (str): Path to raw German audio data
    disease_type (list): List of disease types in German dataset
    voice_type (list): List of voice types to process
    tone_type (list): List of tone types to process

    Returns:
    pd.DataFrame: DataFrame containing German dataset file information
    """

    file_path = []
    p_ids = []
    vtype = []
    ttype = []
    dtype = []
    sample_duration = []
    sample_sr = []

    for disease in disease_type:
        # read all the wav file recursively 
        wav_list = glob.glob(os.path.join(raw_wav_folder, disease) + '/**/*.wav', recursive = True) 

        for wav_file in wav_list:
            temp_v = wav_file.split(os.sep)[-1].split('-')[1].split('_')[0]
            temp_t = wav_file.split(os.sep)[-1].split('-')[-1].split('_')[-1].split('.')[0]
            
            if temp_v in voice_type and temp_t in tone_type:
                file_path.append(wav_file)
                p_ids.append(wav_file.split(os.sep)[-1].split('-')[0])  # id
                vtype.append(temp_v)
                ttype.append(temp_t)
                dtype.append(disease)
                f =  sf.SoundFile(wav_file)
                sr = f.samplerate
                flen = len(f)/f.samplerate
                sample_duration.append(flen)
                sample_sr.append(sr)

    df = pd.DataFrame(list(zip(file_path, p_ids, dtype, vtype, ttype, sample_duration, sample_sr)), 
                           columns =['Reading path', 'ID', 'Disease', 'Voice type','Tone type','Duration','Sample Rate'])

    return df

df_ger = get_all_file_properties_ger(raw_wav_folder_ger, disease_type_ger, voice_type_ger[0:2], tone_type_ger[-1])

save_path = "D:/data/audio/new_SVD_processed_data/svd_original_recording.csv"

os.makedirs(os.path.dirname(save_path), exist_ok=True)
df_ger.to_csv(save_path, index=False, encoding='utf-8-sig')
print(f"Saved to:：{os.path.abspath(save_path)}")

# In[103]:

df_ger.head()

# In[ ]:
def windows(data,window_size,step_size):
    """
    Generate sliding windows for data segmentation.

    Parameters:
    data (array-like): Input data to window
    window_size (int): Size of each window
    step_size (int): Step size between windows

    Yields:
    tuple: Start and end indices of each window
    """

    start = 0
    while start < len(data):
        yield start, start + window_size
        start += int(step_size)


def normalize_voice_len(y,target_len):
    """
    Normalize audio length to target length.

    Parameters:
    y (np.array): Input audio data
    target_len (int): Target length for normalization

    Returns:
    np.array: Normalized audio data with target length
    """

    orignal_len = len(y)
    y = np.reshape(y,[orignal_len,1]).T
    if(orignal_len < target_len):
        res = target_len - orignal_len
        res_data=np.zeros([1,res],dtype=np.float32)
        y = np.reshape(y,[orignal_len,1]).T
        y = np.c_[y,res_data]
    else:
        y = y[:,0:target_len]
    return y[0]

# if hann shape window is not a defaul option
def winfunc(windon_size):
    """
    Generate window function for signal processing.

    Parameters:
    window_size (int): Size of the window

    Returns:
    np.array: Hann window function
    """

    win=scipy.signal.get_window('hann',windon_size)
    return win

# In[95]:
def extract_features_mel(X,sample_rate,window_size, n_coef = 128):
    """
    Extract mel spectrogram features.

    Parameters:
    X (np.array): Audio data
    sample_rate (int): Sample rate of audio
    window_size (int): Window size for FFT
    n_coef (int): Number of mel coefficients

    Returns:
    np.array: Mel spectrogram features
    """
    step_size=window_size//4
    melspec = librosa.feature.melspectrogram(y=X,sr=sample_rate,n_mels=n_coef,n_fft=window_size,hop_length=step_size)
    return melspec[:, :, np.newaxis]

# In[20]:
def extract_features_pitch(soundFile, f0min, f0max, unit):
    """
    Extract pitch-related acoustic parameters including jitter and shimmer.

    Parameters:
    soundFile (str or np.array): Path to audio file or audio data
    f0min (float): Minimum fundamental frequency
    f0max (float): Maximum fundamental frequency
    unit (str): Unit for pitch measurement

    Returns:
    tuple: Various pitch-related features (meanF0, stdevF0, hnr, jitter and shimmer metrics)
    """

    sound = parselmouth.Sound(soundFile) # read the sound
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max) #create a praat pitch object
    meanF0 = call(pitch, "Get mean", 0, 0, unit) # get mean pitch
    stdevF0 = call(pitch, "Get standard deviation", 0 ,0, unit) # get standard deviation
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11Shimmer =  call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    return meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer

# In[96]:
def split_gen_pkl(feature_type, voice_type, df_v, label_dict, lang, target_dur = 1.0, step_dur = 0.6): #only one type of feature extraction at a time
    """
    Core function: Split audio into fixed-length segments, extract specified features (mel or pitch),
    and save results as pickle files. Processed according to different voice types and disease labels
    to ensure each sample has corresponding features and labels. Handles both Chinese and German datasets,
    generating different pickle files.

    Parameters:
    feature_type (str): Type of features to extract ('mel' or 'pitch')
    voice_type (list): List of voice types to process
    df_v (pd.DataFrame): DataFrame containing audio file information
    label_dict (dict): Dictionary mapping disease names to labels
    lang (str): Language identifier ('ch' for Chinese, 'ger' for German)
    target_dur (float): Target duration for each segment
    step_dur (float): Step duration between segments
    """

    function_dict = {'mel': extract_features_mel}
    
    target_sr = 44100
    pkl_save_folder = 'D:/data/audio/final_EENT_processed/pickle_files'  # 中文数据集的新保存路径

    for vtype in voice_type: # what type of voice samples to convert /a/ /i/, /passage/, /chat/
        target_len = int(round(target_sr*target_dur))
        step_len = int(round(target_sr*step_dur))
        ftype=feature_type
         
        df_read = df_v[df_v['Voice type'].str.contains(vtype)].reset_index(drop=True)
        extract_feature = []
        label = []
        ids = []
        
        for i in range(len(df_read)):
        
            read_path = df_read['Reading path'][i]
            y, sr = librosa.load(read_path, sr = target_sr)
            wt = sr*0.03
            window_size = np.power(2,round(math.log2(wt)))
            start = 0
            while start + target_len <= len(y):

                y_split = y[start : (start + target_len)]

                try:
                    if feature_type == 'pitch':
                        pitches= extract_features_pitch(y_split, 75, 500, "Hertz")
                        extract_feature.append(np.array(pitches))
                        label.append(label_dict[df_read['Disease'][i]])
                        ids.append(df_read['ID'][i])

                    else:
                        features = function_dict[ftype](y_split, sr, window_size) # use function dict
                        features = features.astype(np.float64)

                        if features is not None:
                            if (ftype =='6 features mean')|(ftype =='6 features frame'):
                                extract_feature.append(features)

                            else:
                                extract_feature.append(features[np.newaxis,:,:,:])
                            label.append(label_dict[df_read['Disease'][i]])
                            
                            ids.append(df_read['ID'][i])
                except:
                    logging.debug(ftype,read_path)
                    logging.debug(traceback.print_exc())
                    traceback.print_exc()
                start = start + step_len 

        pkl_save_path = os.path.join(pkl_save_folder,'{}_{}_{}.pkl'.format(vtype,ftype,lang))
        dataset={'features':extract_feature,'labels':label, 'ids':ids}
        joblib.dump(dataset,open(pkl_save_path,'wb'))

# In[98]:
# Generate pickle files for Chinese EENT dataset

disease_type = ['Functional Dysphonia','Glottal Incompetence','Nodules','Normal','Polyps','Sulcus', 'VFP']
label_dict = {'Normal': 0, 'Functional Dysphonia': 1, 'Glottal Incompetence':2, 'Nodules':3, 
              'Polyps': 4, 'Sulcus': 5, 'VFP' : 6}
voice_type = ['vowel-a','vowel-i','phrase','passage','chat','mpt']
feature_type = ['mel', 'pitch']

for i in range(len(feature_type)):
    split_gen_pkl(feature_type[i], ['vowel-a', 'vowel-i'], df_split_vowel, label_dict, lang = 'ch', target_dur=1.5, step_dur = 1)

# In[100]:
# Generate pickle files for SVD dataset

raw_wav_folder_ger = r'D:/data/audio/SVD'
disease_type_ger = ['Cordectomy', 'Dysarthrophonia', 'Dysodie', 'Dysphonia', 'Functional dysphonia', 'Healthy',
                    'Hyperfunctional dysphonia', 'Hypofunctional dysphonia', 
                    'Hypotonic dysphonia', 'Laryngitis', 'Polyp', 
                    'Reinke edema', 'Spasmodic dysphonia', 'Uncommon', 'VC carcinoma', 'VFP', ]
label_dict_ger = {'Healthy':0, 'Cordectomy':1, 'Dysarthrophonia':2, 'Dysodie':3, 'Dysphonia':4, 
                  'Functional dysphonia':5, 'Hyperfunctional dysphonia':6, 'Hypofunctional dysphonia':7,
                    'Hypotonic dysphonia':8, 'Laryngitis':9, 'Polyp':10, 
                    'Reinke edema':11, 'Spasmodic dysphonia':12, 'Uncommon':13, 'VC carcinoma':14, 'VFP':15}
voice_type_ger = ['a','i','iau','phrase','u']
tone_type_ger = ['h', 'l', 'lhl', 'n']

feature_type = ['mel', 'pitch']
for i in range(len(feature_type)):
    split_gen_pkl(feature_type[i], ['a', 'i'], df_ger, label_dict_ger, lang='ger', target_dur = 1.5, step_dur = 1)


