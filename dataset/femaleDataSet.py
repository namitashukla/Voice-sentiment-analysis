#labelling of training data
data2_df = data_df.copy()
data2_df = data2_df[data2_df.label != "female_none"]
data2_df = data2_df[data2_df.label != "male_none"].reset_index(drop=True)
data2_df = data2_df[data2_df.label != "male_neutral"]
data2_df = data2_df[data2_df.label != "male_happy"]
data2_df = data2_df[data2_df.label != "male_angry"]
data2_df = data2_df[data2_df.label != "male_sad"]
data2_df = data2_df[data2_df.label != "male_fearful"]
data2_df = data2_df[data2_df.label != "male_calm"]
data2_df = data2_df[data2_df.label != "male_positive"]
data2_df = data2_df[data2_df.label != "male_negative"].reset_index(drop=True)

tmp1 = data2_df[data2_df.actor == 21]
tmp2 = data2_df[data2_df.actor == 22]
tmp3 = data2_df[data2_df.actor == 23]
tmp4 = data2_df[data2_df.actor == 24]

data3_df = pd.concat([tmp1, tmp3],ignore_index=True).reset_index(drop=True)

data2_df = data2_df[data2_df.actor != 21]
data2_df = data2_df[data2_df.actor != 22]
data2_df = data2_df[data2_df.actor != 23].reset_index(drop=True)
data2_df = data2_df[data2_df.actor != 24].reset_index(drop=True)

#output
print (len(data2_df))
data2_df.head()
print (len(data3_df))
data3_df.head()

#extracting mfcc features using librosa
data = pd.DataFrame(columns=['feature'])
for i in tqdm(range(len(data2_df))):
    X, sample_rate = librosa.load(data2_df.path[i], res_type='kaiser_fast',
                                  duration=input_duration,sr=22050*2,offset=0.5)
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
    feature = mfccs
    data.loc[i] = [feature]

#new dataset newdf with 259 features of 400 samples with their corresponding labels
df3 = pd.DataFrame(data['feature'].values.tolist())
labels = data2_df.label
newdf = pd.concat([df3,labels], axis=1)

#setting 0 instead of nan
rnewdf = newdf.rename(index=str, columns={"0": "label"})
rnewdf = rnewdf.fillna(0)
