import numpy as np
import pandas as pd
import re

import nltk
from nltk.corpus import stopwords


from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import KFold
import keras



Jobs_Dataset = pd.read_csv('./fake_job_postings.csv')
jobs_DS = Jobs_Dataset.copy()
jobs_DS= jobs_DS[jobs_DS.columns[jobs_DS.isnull().mean()<0.4]]
jobs_DS.fillna('',inplace=True)
jobs_DS.drop(columns=['telecommuting','has_company_logo','has_questions','job_id'], inplace=True)
jobs_DS['text'] = jobs_DS['title']+" "+jobs_DS['location']+" "+jobs_DS['company_profile']+" "+jobs_DS['description']+" "+jobs_DS['requirements']+" "+jobs_DS['employment_type']+" "+jobs_DS['required_experience']+" "+jobs_DS['industry']+" "+jobs_DS['function']
jobs_DS.drop(columns=['title','location','company_profile','description','requirements','employment_type','required_experience','industry','function'],inplace=True)

jobs_DS['text'] = jobs_DS['text'].str.replace('\n', ' ')
jobs_DS['text'] = jobs_DS['text'].str.replace('\r', ' ')
jobs_DS['text'] = jobs_DS['text'].str.replace('\t', ' ')

jobs_DS['text'] = jobs_DS['text'].apply(lambda x: re.sub(r'[0-9]',' ',x))
jobs_DS['text'] = jobs_DS['text'].apply(lambda x: re.sub(r'[/(){}\[\]\|@,;.:-]',' ',x))

jobs_DS['text']= jobs_DS['text'].apply(lambda s:s.lower() if type(s) == str else s)

jobs_DS['text']= jobs_DS['text'].apply(lambda s:" ".join(s.split()) if type(s) == str else s)

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

jobs_DS['text'] = jobs_DS['text'].apply(lambda x:' '.join([word for word in x.split() if word not in (stop_words)]))

one_hot_x = [one_hot(description,5000) for description in jobs_DS['text']]
max_l = 40
embedded_description = pad_sequences(one_hot_x,max_l)

METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR')
]

model = Sequential()
model.add(Embedding(5000,40,'uniform',input_length=max_l))
model.add(Bidirectional(LSTM(100)))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=METRICS)
print(model.summary())

X = np.array(embedded_description)
Y = np.array(jobs_DS['fraudulent'])

kf = KFold(n_splits=3)

for train_index, test_index in kf.split(X):
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]  

    model.fit(X_train,y_train, validation_data=(X_test,y_test), epochs=6, batch_size=30)

def predict(m,fake_job_post):
  input = fake_job_post.replace('\n',' ').replace('\r',' ').replace('\t',' ')
  input = re.sub(r'[0-9]',' ',input)
  input = re.sub(r'[/(){}\[\]\|@,;.:-]',' ',input)
  input = input.lower()
  input = " ".join(input.split())
  input = ' '.join([word for word in input.split() if word not in (stop_words)])

  one_hot_input = one_hot(input,5000)
  embedded = pad_sequences([one_hot_input],maxlen=max_l)

  pred = m.predict(embedded)
  print(pred)

  if(pred > 0.5): return "This job posting its FAKE"
  return "This job posting its TRUE"   

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Description(BaseModel):
  description: str

@app.post('/predict')
def predict_job_posting(request: Description):
  description =  request.description
  return predict(model,description)