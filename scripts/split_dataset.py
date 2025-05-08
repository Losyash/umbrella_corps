import splitfolders

input_path = 'data/dataset_1'
output_path = 'data/splited/dataset_3'

splitfolders.ratio(input_path, output=output_path, seed=42, ratio=(.7, .2, .1))