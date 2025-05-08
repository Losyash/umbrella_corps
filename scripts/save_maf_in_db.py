from sqlalchemy import create_engine
from src.utils import get_filelist
import pandas as pd


files_path = 'data/raw/tcga/tcga_maf_1'
files = get_filelist(files_path, ext='.csv')

connection_string = 'postgresql+psycopg2://postgres:pwd123qwe@localhost:5432/genome'
db = create_engine(connection_string) 

chromosome_dictionary ={
  'chr1'  : 1,  'chr2'  : 2,  'chr3'  : 3,  'chr4'  : 4,  'chr5'  : 5,  'chr6'  : 6,  
  'chr7'  : 7,  'chr8'  : 8,  'chr9'  : 9,  'chr10' : 10, 'chr11' : 11, 'chr12' : 12,
  'chr13' : 13, 'chr14' : 14, 'chr15' : 15, 'chr16' : 16, 'chr17' : 17, 'chr18' : 18,
  'chr19' : 19, 'chr20' : 20, 'chr21' : 21, 'chr22' : 22, 'chrX'  : 23, 'chrY'  : 24
}

for file in files:
  df = pd.read_csv(f'{files_path}/{file}', sep=';', index_col=0, low_memory=False)
  df['own_chromosome_index'] = df['Chromosome'].map(chromosome_dictionary)
  df.to_sql('dna_all', db, if_exists='append')