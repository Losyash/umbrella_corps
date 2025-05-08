import pandas as pd
from sqlalchemy import create_engine


connection_string = 'postgresql+psycopg2://postgres:pwd123qwe@localhost:5432/genome'
db = create_engine(connection_string)


chromosomes_df = pd.read_sql(
  'SELECT DISTINCT dm."own_chromosome_index" \
  FROM dna_all dm \
  ORDER BY dm."own_chromosome_index"', db
)

for chromosome_id in chromosomes_df['own_chromosome_index'].values:
  genes_list = []
  genes_df = []

  genes = pd.read_sql(
    f'SELECT dm.* \
    FROM dna_all dm \
    WHERE dm."own_chromosome_index" = {chromosome_id} \
    ORDER BY dm."Start_Position"', db
  )

  for index, row in genes.iterrows():
    if row['Hugo_Symbol'] not in genes_list:
      genes_list.append(row['Hugo_Symbol'])
      genes_df.append({ 'Hugo_Symbol': row['Hugo_Symbol'] })

  genes_df = pd.DataFrame(genes_df)
  genes_df.to_sql(f'chromosome_{chromosome_id}', db, if_exists='append')