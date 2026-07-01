# main.py
import pipeline

INPUT_FILE = "data/queries_with_results.json"


def main():
    print("Starting query classification pipeline...")
    result = pipeline.process(INPUT_FILE)
    print(f"Done. Stats: {result}")


if __name__ == "__main__":
    main()
