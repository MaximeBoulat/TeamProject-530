from DataExploration import DataExploration
from DataPreprocessing import DataPreprocessing, ModelType

from example_usage import example_traditional_ml, example_lstm, example_neural_network


def main():

    data_preprocessor = DataPreprocessing()
    df = data_preprocessor.preprocess_data(model_type=ModelType.NEURAL_NETWORK)

    print(df.head())

    # example_traditional_ml(df)

    # example_lstm(df)

    example_neural_network(df)
    

    input("Press Enter to exit...")
    print("end of program")

    return 0


if __name__ == "__main__":
    exit(main()) 