from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import cv2
import numpy as np
import os
import pandas as pd
import requests


gds_endpoint = 'https://api.gdc.cancer.gov/data'


# Скачивание файла ------------------------------------------------------------------>
def download(url, save_dir):
  response = requests.get(url, stream=True, allow_redirects=True)
  file_name = response.headers.get('content-disposition').split('filename=')[1]
  total_size = int(response.headers.get('content-length', 0))

  with open(f'{save_dir}\\{file_name}', 'wb') as file, tqdm(
    desc=f'{save_dir}\\{file_name}',
    total=total_size,
    unit='B',
    unit_scale=True,
    unit_divisor=1024,
    ascii=' ='
  ) as bar:
    for data in response.iter_content(chunk_size=1024):
      size = file.write(data)
      bar.update(size)
# <------------------------------------------------------------------ Скачивание файла


# Получение списка файлов в каталоге ------------------------------------------------>
def get_filelist(path, ext='.*'):
  return [f for f in listdir(path) if isfile(join(path, f)) and f.endswith(ext)]
# <------------------------------------------------ Получение списка файлов в каталоге


# Загрузка данных с портала GDC ----------------------------------------------------->
def get_from_gdc(file, save_dir, field='File UUID'):
  data = pd.read_csv(file, sep='\t')

  for l in data[field]:
    download(f'{gds_endpoint}/{l}', save_dir)
# <----------------------------------------------------- Загрузка данных с портала GDC


# Объединить данные из MAF файлов --------------------------------------------------->
def concat_maf_files(path, files, cancer_type):
  dfs = []
  dfs_length = 0

  for file in files:
    df = pd.read_csv(f'{path}/{file}', sep='\t', skiprows=7, index_col=0, header=0)
    df['Cancer_Type_Own'] = cancer_type
    df['File_Name_Own'] = file
    dfs.append(df)

    dfsl += len(df)

  print(len(pd.concat(dfs, axis=0)), dfs_length)
  return pd.concat(dfs, axis=0)
# <--------------------------------------------------- Объединить данные из MAF файлов


# Загрузка данных для модели ---------------------------------------------------------->
def load_data(input_path, image_width, image_height):
  label = -1

  cancer_data = []
  cancer_labels = []
  cancer_classes = []

  cancer_type_folders = [ f for f in os.listdir(input_path) if os.path.isdir(f'{input_path}/{f}') ]

  for cancer_type_folder in cancer_type_folders:
    cancer_samples_folder = f'{input_path}/{cancer_type_folder}'
    cancer_classes.append(cancer_type_folder)

    label = label + 1

    cancer_sample_files = get_filelist(cancer_samples_folder, '.png')

    for cancer_sample_file in cancer_sample_files:
      cancer_sample_image = f'{cancer_samples_folder}/{cancer_sample_file}'

      image = cv2.imread(cancer_sample_image)
      image = cv2.resize(image,(image_width, image_height))
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

      cancer_data.append(image)
      cancer_labels.append(label)
  
  return np.array(cancer_data), np.array(cancer_labels), cancer_classes
# <---------------------------------------------------------- Загрузка данных для модели