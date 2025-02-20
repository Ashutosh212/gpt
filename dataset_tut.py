import torch
from pathlib import Path
from config import get_config, get_weights_file_path, latest_weights_file_path


from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

def get_all_sentences(ds, lang):
    for item in ds:
        # Skip the pair if any sentence is empty
        if not item.get(lang):  # If the sentence is empty (None or empty string)
            continue
        yield item[lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='unk'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_token = ["unk", "pad", "sos", "eos"], min_freequency = 2)
        # tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        sentences = list(get_all_sentences(ds, lang))
        tokenizer.train_from_iterator(sentences, trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

def get_ds(config):
    ds_raw = load_dataset("csv" , data_files = "hindi_english_parallel.csv")

    ds_raw = ds_raw['train']
    dataset = [{"english": example["english"], "hindi": example["hindi"]} for example in ds_raw]
    # dataset = dataset[:100]
    print(f"len of dataset: {len(dataset)}")

    
    # for example in ds_raw:
    #     hindi_sentence = example['hindi']  # Replace 'hindi' with the actual column name
    #     english_sentence = example['english']  # Replace 'english' with the actual column name
        
    #     # Print the pair of sentences
    #     print("Hindi: ", hindi_sentence)
    #     print("English: ", english_sentence)
    #     print()  # Add an empty line between pairs for clarity
    #     break  # Only print the first pair, remove 'break' to print all pairs
    
    
    # dataset = [{"en": "Hello", "fr": "Bonjour"}, {"en": "World", "fr": "Monde"}]
    print(type(dataset[0]['english']))
    print(type(dataset))
    # print(dataset)
#     print(ds_raw)
#     print("="*50)
# Get the generator
    # gen = get_all_sentences(dataset, "english")

    # print(type(gen))  # Expected: <class 'generator'>
    # print(next(gen))  # Expected: "Hello"
#     print(next(gen))  # Expected: "World"
    # print(next(gen))  # Expected: "Hello"
    # get_all_sentences(ds_raw, config["lang_src"])
    tokenizer_src = get_or_build_tokenizer(config, dataset, "english")
    # tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])

    # return tokenizer_src
config = get_config()
get_ds(config)




