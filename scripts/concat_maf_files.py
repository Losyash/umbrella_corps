from src.utils import get_filelist, concat_maf_files


files_path = 'data/raw/tcga/tcga_maf_1/tcga_uvm'
cancer_type = 'UVM'

files = get_filelist(files_path, ext='.maf')


df  = concat_maf_files(files_path, files, cancer_type)
df.to_csv(f'{files_path}/{cancer_type.lower()}.aliquot_ensemble_masked.csv', sep=';')