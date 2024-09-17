import csv
import tiktoken
import tqdm


def tokenize(input_file, output_file):
    '''
    Takes a csv file path as input and tokenizes data and saves it to output file.
    '''
    tokenizer = tiktoken.get_encoding("gpt2")
    END_TOKEN = "<|endoftext|>"
    all_tokens = []
    with open(input_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in tqdm.tqdm(reader, desc='Tokenizing Data'):
            text = row['text']
            text_with_end = text + END_TOKEN
            tokens = tokenizer.encode(text_with_end, allowed_special={END_TOKEN})
            all_tokens.extend(tokens)
    
    token_string = " ".join(map(str, all_tokens))
    
    with open(output_file, 'w') as f:
        f.write(token_string)

    print(f"Processed {len(all_tokens)} tokens in total.")
    print(f"Tokens saved to {output_file}")


def main():
    input_train_file = 'data/tiny-stories/train.csv'
    output_train_file = 'data/tiny-stories/train_tokens.txt'
    input_validation_file = 'data/tiny-stories/validation.csv'
    output_validation_file = 'data/tiny-stories/validation_tokens.txt'

    tokenize(input_file=input_train_file, output_file=output_train_file)
    tokenize(input_file=input_validation_file, output_file=output_validation_file)
    

if __name__ == '__main__':
    main()
