import os
import subprocess

def run_step(description, command):
    print("\n=================================================")
    print(f"▶️  {description}")
    print("=================================================\n")

    process = subprocess.run(command, shell=True)
    if process.returncode != 0:
        print(f"❌ FAILED at step: {description}")
        exit(1)

    print(f"✔ COMPLETED: {description}\n")


def main():

    # 1. Generate raw data
    run_step(
        "Generating raw fruit & vegetable datasets",
        "python3 Raw_Data/generate_data.py"
    )

    # 2. Preprocess raw data → clean data
    run_step(
        "Cleaning raw data",
        "python3 Data_preprocessing/data_preprocessing.py"
    )

    # 3. Feature engineering → X.csv, y.csv, scalers, encoders
    run_step(
        "Preparing ML features",
        "python3 Data_preprocessing/prepare_features.py"
    )

    # 4. Train all models (itemwise + global + moving average)
    run_step(
        "Training all models",
        "python3 model/train_models.py"
    )

    # 5. Predict next 7 days
    run_step(
        "Predicting next 7 days demand",
        "python3 model/predict_next_7_days.py"
    )

    print("\n🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
    print("You can now check Model_Data/ for predictions output.\n")


if __name__ == "__main__":
    main()
