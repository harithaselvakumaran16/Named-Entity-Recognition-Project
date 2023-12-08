import argparse
import NER_functions


def main():
    # Define the command-line arguments
    parser = argparse.ArgumentParser(description="NER Model")

    # Add an argument for the path to the NER model
    parser.add_argument("--model_path", type=str, default="/content/ner_model.pth", help="Path to the NER model")

    # Parse the command-line arguments
    args = parser.parse_args()

    model_path = "Model/ner_model.pth"

    while True:
        print("Welcome to Advanced Named Entity Recognition!!!")
        print("\nMain Menu:")
        print("1. Input text sentence")
        print("2. Input audio file")
        print("3. Exit")

        choice = input("Enter your choice (1/2/3): ")

        if choice == "1":
            test_sentence = input("Enter the text sentence: ")
            NER_functions.predict_entities(test_sentence, model_path)

        elif choice == "2":
            file_path = input("Enter the path of audio file: ")
            text = NER_functions.audio_to_text(file_path)
            NER_functions.predict_entities(text, model_path)

        elif choice == "3":
            print("Exiting the program. Goodbye!")
            break

        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
