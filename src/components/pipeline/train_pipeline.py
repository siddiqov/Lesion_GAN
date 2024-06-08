from src.components.data_ingestion import create_dataset_structure
from src.components.data_transformation import prepare_data
from src.components.model_trainer.py import train_model
import os

def run_training_pipeline():
    create_dataset_structure()
    (train_data, train_labels), (test_data, test_labels) = prepare_data()
    model, label_encoder = train_model(train_data, train_labels)

    model_save_path = 'artifacts/models/lesion_classifier.h5'
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    return model, label_encoder

if __name__ == "__main__":
    run_training_pipeline()
