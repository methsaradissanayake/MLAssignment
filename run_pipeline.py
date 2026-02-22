import argparse
from pipeline.data_loader import DataLoader
from pipeline.preprocessor import Preprocessor
from pipeline.model_builder import XGBModelBuilder
from pipeline.model_evaluator import Evaluator

def main(input_csv):
    print("="*40)
    print("1. LOADING DATA")
    print("="*40)
    loader = DataLoader(input_csv)
    df = loader.load_data()
    df_cleaned = loader.clean_target()
    
    print("\n" + "="*40)
    print("2. PREPROCESSING")
    print("="*40)
    preprocessor = Preprocessor(df_cleaned, loader.target_col)
    preprocessor.extract_mappings()
    preprocessor.select_features()
    preprocessor.split_and_save(output_dir="data", test_size=0.15, val_size=0.15)
    
    print("\n" + "="*40)
    print("3. MODEL TRAINING & TUNING")
    print("="*40)
    trainer = XGBModelBuilder(data_dir="data", models_dir="models")
    trainer.tune_and_train(n_iter=15)
    trainer.save_model()
    
    print("\n" + "="*40)
    print("4. MODEL EVALUATION & SHAP")
    print("="*40)
    evaluator = Evaluator(data_dir="data", models_dir="models", outputs_dir="outputs")
    evaluator.evaluate_performance()
    evaluator.generate_plots()
    evaluator.generate_shap()
    
    print("\n" + "="*40)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run the full OOP ML Pipeline.")
    parser.add_argument("--input", type=str, required=True, help="Input raw CSV mapped from scraper")
    args = parser.parse_args()
    main(args.input)
