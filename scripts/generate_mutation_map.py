from sqlalchemy import create_engine
import cv2
import numpy as np
import os
import pandas as pd


# Подключение к БД ------------------------------------------------------------------>
connection_string = 'postgresql+psycopg2://postgres:pwd123qwe@localhost:5432/genome'
db = create_engine(connection_string)
# <------------------------------------------------------------------ Подключение к БД


# Загрузка списка хромосом ---------------------------------------------------------->
chromosome_ids_df = pd.read_sql(
  'SELECT DISTINCT dm."Chromosome_Index_Own" \
  FROM dna_all dm \
  ORDER BY dm."Chromosome_Index_Own"', db
)

chromosome_ids = chromosome_ids_df['Chromosome_Index_Own'].values
# <---------------------------------------------------------- Загрузка списка хромосом


# Загрузка списка типов рака -------------------------------------------------------->
cancer_types_df = pd.read_sql(
  'SELECT DISTINCT dm."Cancer_Type_Own" \
  FROM dna_all dm \
  ORDER BY dm."Cancer_Type_Own"', db
)

cancer_types = cancer_types_df['Cancer_Type_Own'].values
# <-------------------------------------------------------- Загрузка списка типов рака


# Загрузка списка хромосом и формирование справочников ------------------------------>
gene_chromosomes = {}
gene_indexes = {}

for chromosome_id in chromosome_ids:
  chromosome_genes = pd.read_sql(
    f'SELECT cr.* \
    FROM chromosome_{chromosome_id} cr \
    ORDER BY cr."Index"', db
  )

  genes_by_chromosomes = []
  genes_by_index = {}

  for index, row in chromosome_genes.iterrows():
    genes_by_chromosomes.append(row['Hugo_Symbol'])
    genes_by_index[row['Hugo_Symbol']] = row['Index']

  gene_chromosomes[chromosome_id - 1] = genes_by_chromosomes
  gene_indexes[chromosome_id - 1] = genes_by_index

  # Для проверки -------------------------------------------------------------------->
  # cv2.imwrite(f'{chromosome_id}.png', gene_arrays[chromosome_id - 1])

  # with open('gene_chromosomes.json', 'a') as f:
  #   f.write(json.dumps(gene_chromosomes))

  # with open('gene_indexes.json', 'a') as f:
  #   f.write(json.dumps(gene_indexes))
  # <-------------------------------------------------------------------- Для проверки
# <------------------------------ Загрузка списка хромосом и формирование справочников


# Генерация карты мутаций ----------------------------------------------------------->
def create_gene_arrays(chromosome_ids, gene_chromosomes, matrix_n):
  gene_arrays = {}

  for chromosome_id in chromosome_ids:
    gene_count = len(gene_chromosomes[chromosome_id - 1])
    gene_width = gene_count // 261

    if gene_count % 261 != 0:
      gene_width += 1

    gene_arrays[chromosome_id - 1] = np.zeros([ 261, gene_width * 3, 3 ], np.uint8) + 255

  return gene_arrays
# <----------------------------------------------------------- Генерация карты мутаций


# Генерация карты мутаций ----------------------------------------------------------->
for cancer_type in cancer_types:
  dataset_path = f'data/dataset_2/{cancer_type.lower()}'

  if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

  # gene_arrays = create_gene_arrays(chromosome_ids, gene_chromosomes, 261)

  tumor_samples_df = pd.read_sql(
    f'SELECT DISTINCT dm."Tumor_Sample_Barcode" \
    FROM dna_all dm \
    WHERE dm."Cancer_Type_Own" = \'{cancer_type}\' \
    ORDER BY dm."Tumor_Sample_Barcode"', db
  )

  tumor_samples = tumor_samples_df['Tumor_Sample_Barcode'].values

  for i, tumor_sample in enumerate(tumor_samples):
    tumor_sample_df = pd.read_sql(
      f'SELECT DISTINCT dm.* \
      FROM dna_all dm \
      WHERE dm."Tumor_Sample_Barcode" = \'{tumor_sample}\' AND dm."Variant_Type" IN (\'SNP\', \'INS\', \'DEL\') \
      ORDER BY dm."Start_Position"', db
    )

    gene_arrays = create_gene_arrays(chromosome_ids, gene_chromosomes, 261)
    data = np.zeros([ 261, 261, 3 ], np.uint8) + 255

    snp_num = 0
    ins_num = 0
    del_num = 0

    for index, row in tumor_sample_df.iterrows():
      gene_name = row['Hugo_Symbol']
      mutation_type = row['Variant_Type']
      chromosome_id = row['own_chromosome_index']

      gene_index = gene_indexes[chromosome_id - 1][gene_name]

      if gene_index < 261:
        idx = gene_index
        jdx = 0
      else:
        idx = gene_index % 261
        jdx = (gene_index // 261) * 3

      # Для проверки -------------------------------------------------------------------->
      # print(f'{tumor_sample} {chromosome_id - 1} {gene_name} {gene_index}, {idx}, {jdx}')
      # <-------------------------------------------------------------------- Для проверки

      if (mutation_type == 'SNP'):
        gene_arrays[chromosome_id - 1][idx, jdx] = [255, 0, 0]
        snp_num += 1
      elif (mutation_type == 'INS'):
        gene_arrays[chromosome_id - 1][idx, jdx + 1] = [0, 255, 0]
        ins_num += 1
      elif (mutation_type == 'DEL'):
        gene_arrays[chromosome_id - 1][idx, jdx + 2] = [0, 0, 255]
        del_num += 1

      # Для проверки -------------------------------------------------------------------->
      # cv2.imwrite(f'{tumor_sample}_{chromosome_id}_{cancer_type}.png', gene_arrays[chromosome_id - 1])
      # Для проверки -------------------------------------------------------------------->

    p = 0
    q = 0

    for chromosome_id in chromosome_ids:
      gene_num = len(gene_chromosomes[chromosome_id - 1])

      q = (gene_num // 261) * 3

      if gene_num % 261 != 0:
        q = q + 3

      if p + q <= 261:
        data[: ,p : (p + q)] = gene_arrays[chromosome_id - 1]
        p = p + q
      else:
        print(f'The number of genes is over 261 lines. Data shape {data[:, p : (p + q)].shape}')

    n = i + 1 if (i + 1 > 10) else f'0{i + 1}'

    img_path = f'{dataset_path}/{n}_{tumor_sample}.png'
    cv2.imwrite(img_path, data)
# <----------------------------------------------------------- Генерация карты мутаций