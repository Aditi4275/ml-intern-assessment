from ngram_model import TrigramModel

def main():
    # Create a new TrigramModel
    model = TrigramModel()

    # Train the model on the example corpus
    with open("data/tale_of_two_cities.txt", "r") as f:
        text = f.read()
    model.fit(text)

    # Generate new text
    generated_text = model.generate()
    print("Generated Text:")
    print(generated_text)

if __name__ == "__main__":
    main()
