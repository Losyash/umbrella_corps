from confidenceinterval import (
  classification_report_with_ci,
  precision_score,
  recall_score,
  f1_score
)
from confidenceinterval.bootstrap import bootstrap_ci
from datetime import datetime
from keras.utils import to_categorical
import keras
import numpy as np
import pandas as pd
import sklearn
import tensorflow.keras.models
from src.utils import load_data

model = keras.saving.load_model('models/xception_dataset_2_04.05.2025-20.45.keras')

def main():
  IMAGE_SIZE = 261
  N_CLASSES = 33

  BASE_MODEL = 'xception'

  # model_path = 'models'
  history_path = 'history'

  dataset = 'dataset_2'
  dataset_path = f'data/splited/{dataset}'
  dataset_test_path = f'{dataset_path}/test'

  X_test, y_test, classes_test = load_data(dataset_test_path, IMAGE_SIZE, IMAGE_SIZE)

  X_test = X_test.astype('float32') / 255
  y_test = to_categorical(y_test, N_CLASSES)

  y_pred = model.predict(X_test)

  y_test_list = np.argmax(y_test, axis=1)
  y_pred_list = np.argmax(y_pred, axis=1)

  # loss = keras.losses.mean_squared_error(y_test_list, y_pred_list)

  report_scores = classification_report_with_ci(y_test_list, y_pred_list, round_ndigits=4)

  report_df = pd.DataFrame(report_scores)
  report_df.to_csv(
    f'{history_path}/{BASE_MODEL}/{dataset}/report_w_ci_{dataset}_{datetime.now().strftime("%d.%m.%Y-%H.%M")}.csv',
    # index=False
  )

if __name__ == "__main__":
    main()