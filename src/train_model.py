import pandas as pd
import numpy as np
import os
from datetime import datetime
from keras.applications import (
  InceptionV3,
  InceptionResNetV2,
  ResNet50V2,
  ResNet152V2,
  Xception,
  DenseNet201
)
from keras.callbacks import (
  Callback,
  ReduceLROnPlateau,
  EarlyStopping,
  ModelCheckpoint,
  CSVLogger
)
from keras.metrics import (
  FalseNegatives,
  FalsePositives,
  TrueNegatives,
  TruePositives,
  Precision,
  Recall,
  F1Score,
  FBetaScore
)
from keras.optimizers import SGD
from keras.utils import to_categorical
from sklearn.metrics import (
  accuracy_score,
  classification_report,
  confusion_matrix,
  f1_score,
  precision_score,
  recall_score
)
from src.utils import load_data

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def main():
  N_BATCHES = 8
  N_CLASSES = 33
  N_EPOCHS = 100
  IMAGE_SIZE = 261

  # inception_v3, inception_resnet_v2, resnet_50_v2, resnet_152_v2
  # xception, dense_net_201
  BASE_MODEL = 'dense_net_201'

  input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)

  dataset = 'dataset_2'
  dataset_path = f'data/splited/{dataset}'
  dataset_train_path = f'{dataset_path}/train'
  dataset_test_path = f'{dataset_path}/test'
  dataset_valid_path = f'{dataset_path}/val'

  model_path = 'models'
  history_path = 'history'

  X_train, y_train, classes_train = load_data(dataset_train_path, IMAGE_SIZE, IMAGE_SIZE)
  X_test, y_test, classes_test = load_data(dataset_test_path, IMAGE_SIZE, IMAGE_SIZE)
  X_valid, y_valid, classes_valid = load_data(dataset_valid_path, IMAGE_SIZE, IMAGE_SIZE)

  input_shape = X_train.shape[1:]

  # Нормализация данных --------------------------------------------------------------->
  X_train = X_train.astype('float32') / 255
  X_test = X_test.astype('float32') / 255
  X_valid = X_valid.astype('float32') / 255
  # <--------------------------------------------------------------- Нормализация данных

  print(
    f'Количество образцов для обучения     | {X_train.shape[0]}',
    f'Количество образцов для тестирования | {X_test.shape[0]}',
    f'Количество образцов для проверки     | {X_valid.shape[0]}',
    sep='\n'
  )

  # print(f'Размерность образцов для обучения | {X_train.shape} | {y_train.shape}')

  y_train = to_categorical(y_train, N_CLASSES)
  y_test = to_categorical(y_test, N_CLASSES)
  y_valid = to_categorical(y_valid, N_CLASSES)

  metrics = [
    FalseNegatives(name="fn"),
    FalsePositives(name="fp"),
    TrueNegatives(name="tn"),
    TruePositives(name="tp"),
    Precision(name="precision"),
    Recall(name="recall"),
    F1Score(name='f1_macro', average='macro'),
    F1Score(name='f1_micro', average='micro'),
    FBetaScore(name='f1_beta_macro', average='macro'),
    FBetaScore(name='f1_beta_micro', average='micro'),
    'accuracy'
  ]

  reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    verbose=1,
    factor=0.5,
    patience=3,
    min_lr=0.00001
  )

  early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5
  )

  csv_logger = CSVLogger(
    filename=f'{history_path}/{BASE_MODEL}/{dataset}/metrics_{dataset}_{datetime.now().strftime("%d.%m.%Y-%H.%M")}.csv',
    separator=';',
    append=True
  )

  # model_checkpoint = ModelCheckpoint(
  #   filepath=f'{model_path}/{BASE_MODEL}/{BASE_MODEL}_' + '{epoch:003d}.h5',
  #   # save_weights_only = True,
  #   verbose=1 
  # )

  # class SaveModel(Callback):
  #   def on_epoch_end(self, epoch, logs=None):
  #     self.model.save(f'{model_path}/{BASE_MODEL}/{BASE_MODEL}_{epoch}.h5')

  # save_model = SaveModel()

  # class SaveHistory(Callback):
  #   def on_epoch_end(self, epoch, logs=None):
  #     if not ('history.csv' in os.listdir(history_path)):
  #       with open(f'{history_path}/history.csv', 'a') as f:
  #         content = csv.DictWriter(f, logs.keys())
  #         content.writeheader()

  #     with open(f'{history_path}/history.csv','a') as f:
  #       content = csv.DictWriter(f, logs.keys())
  #       content.writerow(logs)

  # save_history = SaveHistory()

  if BASE_MODEL == 'inception_v3':
    model = InceptionV3(
      input_shape=input_shape,
      classes=N_CLASSES,
      weights=None,
      include_top=True
    )

  if BASE_MODEL == 'inception_resnet_v2':
    model = InceptionResNetV2(
      include_top=True,
      input_shape=input_shape,
      classes=N_CLASSES,
      weights=None
    )

  if BASE_MODEL == 'resnet_50_v2':
    model = ResNet50V2(
      input_shape=input_shape,
      classes=N_CLASSES,
      weights=None,
      include_top=True
    )

  if BASE_MODEL == 'resnet_152_v2':
    model = ResNet152V2(
      include_top=True,
      input_shape=input_shape,
      classes=N_CLASSES,
      weights=None
    )

  if BASE_MODEL == 'xception':
    model = Xception(
      include_top=True,
      input_shape=input_shape,
      classes=N_CLASSES,
      weights=None
    )

  if BASE_MODEL == 'dense_net_201':
    model = DenseNet201(
      include_top=True,
      input_shape=input_shape,
      classes=N_CLASSES,
      weights=None
    )

  optimizer = SGD(learning_rate=1e-3)

  model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=metrics
  )

  model.summary()

  history_scores = model.fit(
    X_train, y_train,
    batch_size=N_BATCHES,
    validation_data=(X_valid, y_valid),
    epochs=N_EPOCHS,
    verbose=1,
    callbacks=[reduce_lr, early_stopping, csv_logger]
  )

  # Сохранение модели ----------------------------------------------------------------->
  model.save(f'{model_path}/{BASE_MODEL}_{dataset}_{datetime.now().strftime("%d.%m.%Y-%H.%M")}.keras')
  # <----------------------------------------------------------------- Сохранение модели

  history_df = pd.DataFrame(history_scores.history)
  history_df.to_csv(
    f'{history_path}/{BASE_MODEL}/{dataset}/history_{dataset}_{datetime.now().strftime("%d.%m.%Y-%H.%M")}.csv',
    # index=False
  )

  # Оценка модели на тестовых данных -------------------------------------------------->
  eval_scores = model.evaluate(X_test, y_test, verbose=1, return_dict=True)

  eval_df = pd.DataFrame(eval_scores, index=[0])
  eval_df.to_csv(
    f'{history_path}/{BASE_MODEL}/{dataset}/evaluate_{dataset}_{datetime.now().strftime("%d.%m.%Y-%H.%M")}.csv',
    # index=False
  )
  # <-------------------------------------------------- Оценка модели на тестовых данных

  # Оценка модели на тестовых данных -------------------------------------------------->
  y_pred = model.predict(X_test)

  y_test_list = np.argmax(y_test, axis=1)
  y_pred_list = np.argmax(y_pred, axis=1)

  accuracy = accuracy_score(y_test_list, y_pred_list)

  precision_micro = precision_score(y_test_list, y_pred_list, average='micro')
  precision_macro = precision_score(y_test_list, y_pred_list, average='macro')
  precision_weighted = precision_score(y_test_list, y_pred_list, average='weighted')

  recall_micro = recall_score(y_test_list, y_pred_list, average='micro')
  recall_macro = recall_score(y_test_list, y_pred_list, average='macro')
  recall_weighted = recall_score(y_test_list, y_pred_list, average='weighted')

  f1_micro = f1_score(y_test_list, y_pred_list, average='micro')
  f1_macro = f1_score(y_test_list, y_pred_list, average='macro')
  f1_weighted = f1_score(y_test_list, y_pred_list, average='weighted')

  pred_columns = [ 
    'accuracy',
    'precision_micro', 'precision_macro', 'precision_weighted',
    'recall_micro', 'recall_macro', 'recall_weighted',
    'f1_micro', 'f1_macro', 'f1_weighted'
  ]

  pred_values = [[ 
    accuracy,
    precision_micro, precision_macro, precision_weighted,
    recall_micro, recall_macro, recall_weighted,
    f1_micro, f1_macro, f1_weighted
  ]]

  pred_scores = pd.DataFrame(pred_values, columns=pred_columns)
  pred_scores.to_csv(
    f'{history_path}/{BASE_MODEL}/{dataset}/predict_{dataset}_{datetime.now().strftime("%d.%m.%Y-%H.%M")}.csv',
    # index=False
  )
  # <-------------------------------------------------- Оценка модели на тестовых данных
  
  # Отчет по классификации ------------------------------------------------------------>
  report_scores = classification_report(y_test_list, y_pred_list, digits=4, output_dict=True)

  report_df = pd.DataFrame(report_scores).transpose()
  report_df.to_csv(
    f'{history_path}/{BASE_MODEL}/{dataset}/report_{dataset}_{datetime.now().strftime("%d.%m.%Y-%H.%M")}.csv',
    # index=False
  )
  # <------------------------------------------------------------ Отчет по классификации

  # Матрица ошибок -------------------------------------------------------------------->
  matrix_scores = confusion_matrix(y_test_list, y_pred_list)
  
  matrix_scores = pd.DataFrame(matrix_scores).transpose()
  matrix_scores.to_csv(
    f'{history_path}/{BASE_MODEL}/{dataset}/matrix_{dataset}_{datetime.now().strftime("%d.%m.%Y-%H.%M")}.csv',
    # index=False
  )
  # <-------------------------------------------------------------------- Матрица ошибок

if __name__ == "__main__":
    main()