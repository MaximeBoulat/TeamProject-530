from DataExploration import DataExploration
from DataPreprocessing import DataPreprocessing

from example_usage import example_traditional_ml, example_lstm


def main():

    data_preprocessor = DataPreprocessing()
    df = data_preprocessor.preprocess_data()

    print(df.head())

    # example_traditional_ml(df)

    example_lstm(df)

    input("Press Enter to exit...")
    print("end of program")

    return 0


if __name__ == "__main__":
    exit(main()) 