import pandas as pd
import numpy as np
import argparse
import os
import subprocess
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Instalar TensorFlow y NumPy si no están instalados
try:
    import tensorflow as tf
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow", "numpy"])
    import tensorflow as tf
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

try:
    from tqdm import tqdm
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
    from tqdm import tqdm

try:
    import cv2
except ImportError:
   ([sys.executable, "-m", "pip", "install", "opencv-python-headless"])
    import cv2

def assign_label(img,DOG_emotion_type):
    return DOG_emotion_type

def make_train_data(DOG_emotion_type,DIR):
    for img in tqdm(os.listdir(DIR)):
        label=assign_label(img,DOG_emotion_type)
        path = os.path.join(DIR,img)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))

        X.append(np.array(img))
        Z.append(str(label))

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='/opt/ml/processing/input')
    parser.add_argument('--output_path', type=str, default='/opt/ml/processing/output')
    return parser.parse_args()

if __name__ == "__main__":

    args = _parse_args()

    X=[]
    Z=[]
    IMG_SIZE=150
       # Directorios de emociones de perros
    DOG_ANGRY_DIR = os.path.join(args.input_path, 'angry')
    DOG_HAPPY_DIR = os.path.join(args.input_path, 'happy')
    DOG_RELAXED_DIR = os.path.join(args.input_path, 'relaxed')
    DOG_SAD_DIR = os.path.join(args.input_path, 'sad')


    # Procesar datos
    make_train_data('angry', DOG_ANGRY_DIR)
    make_train_data('happy', DOG_HAPPY_DIR)
    make_train_data('relaxed', DOG_RELAXED_DIR)
    make_train_data('sad', DOG_SAD_DIR)

     # Guardar los datos procesados
    output_file = os.path.join(args.output_path, 'processed_data.npz')
    np.savez(output_file, X=X, Z=Z)
    print(f"Datos procesados guardados en {output_file}")
    
    le=LabelEncoder()
    yi=le.fit_transform(Z)
    Y=to_categorical(yi,5)
    X=np.array(X)
    X=X/255

   # Dividir los datos en conjuntos de entrenamiento, validación y prueba
    x_train, x_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.25, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)
    
     
    # Guardar los conjuntos de datos
    np.savez(os.path.join(args.output_path, 'train_data.npz'), x_train=x_train, y_train=y_train)
    np.savez(os.path.join(args.output_path, 'val_data.npz'), x_val=x_val, y_val=y_val)
    np.savez(os.path.join(args.output_path, 'test_data.npz'), x_test=x_test, y_test=y_test)
    print(f"Conjuntos de datos guardados en {args.output_path}")

    np.random.seed(42)
    tf.random.set_seed(42)

    batch_size = 8
    epochs = 35

# # Definir el modelo
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='Same', activation='relu', input_shape=(150,150,3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=96, kernel_size=(3,3), padding='Same', activation='relu', kernel_regularizer=l2(0.001)))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='Same', activation='relu', kernel_regularizer=l2(0.001)))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='Same', activation='relu', kernel_regularizer=l2(0.001)))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))
    model.summary()



   # Configurar el generador de datos
    datagen = ImageDataGenerator(
        rotation_range=40,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        shear_range=0.2,
        fill_mode='nearest'
    )
    datagen.fit(x_train)

    mode
    # Compilar el modelo
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Configurar callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    
     # Entrenar el modelo
    History = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        verbose=1,
        steps_per_epoch=x_train.shape[0] // batch_size,
        callbacks=[early_stopping, reduce_lr]
    )

     # Evaluar el modelo
    pred = model.predict(x_test)
    pred_digits = np.argmax(pred, axis=1)

    print(accuracy_score(np.argmax(y_test, axis=1), pred_digits))
    print(confusion_matrix(np.argmax(y_test, axis=1), pred_digits))
    print(classification_report(np.argmax(y_test, axis=1), pred_digits))
