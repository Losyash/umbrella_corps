import keras

model_path = 'models/resnet_50_v2_dataset_1_04.05.2025-17.10.keras'

loaded_model = keras.saving.load_model(model_path)
loaded_model.summary()