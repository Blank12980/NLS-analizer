import pandas as pd
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
# work with data
with open('./src/dataset.txt','r',encoding='utf-8') as file:
    text = file.readlines() # Определяем стоп-слова

for i in range(len(text)):
    text[i] = text[i].strip().split(';') if len(text[i].strip().split(';')) > 1 else None

try: text.remove(None)
except: ''


def preprocess_text(text):
    stop_words = set(stopwords.words('english')) 
    lemmatizer = WordNetLemmatizer()
    text = text.lower()  # Приводим к нижнему регистру
    text = re.sub(r'[^\w\s]', '', text)  # Удаляем знаки препинания # Определяем стоп-слова
    text = ' '.join(word for word in text.split() if word not in stop_words)  # Удаляем стоп-слова
    lemmatizer = WordNetLemmatizer()
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split())  # Лемматизация
    return text
data =  [{"text": preprocess_text(i[0]),  "emotion": i[1]  }for i in text]
df = pd.DataFrame(data)

# tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df["text"])
X = tokenizer.texts_to_sequences(df["text"])
maxlen = 10000
X = pad_sequences(X,maxlen=maxlen, padding='post')

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["emotion"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=2, random_state=192)

# create NLP
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=maxlen))
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=256, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# start the learn
model.fit(X_train, y_train, epochs=5, batch_size=4, validation_data=(X_test, y_test),shuffle=True)

# save part
model.save('emotion_recognition_model.h5')
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("Модель сохранена в файл 'emotion_recognition_model.h5'.")

