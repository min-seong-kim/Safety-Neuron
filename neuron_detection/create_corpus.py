import os
from datasets import load_dataset

def main():
    # Load the dataset from Hugging Face
    print("Loading dataset 'LibrAI/do-not-answer' train split...")
    dataset = load_dataset("LibrAI/do-not-answer", split="train")
    
    # Extract the 'question' field from each item
    questions = [item['question'] for item in dataset]
    print(f"Extracted {len(questions)} questions.")
    
    # Create corpus_all directory if it doesn't exist
    os.makedirs("./corpus_all", exist_ok=True)
    
    # Write questions to a new file in corpus_all
    output_file = "./corpus_all/do_not_answer.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for question in questions:
            f.write(question + "\n")
    
    print(f"Questions saved to {output_file}")

if __name__ == "__main__":
    main()