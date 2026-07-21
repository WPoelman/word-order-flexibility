from datasets import load_dataset
from collections import defaultdict
import subprocess
from tqdm import tqdm
import pandas as pd
import json
from qq import Database, IdType 

ALL_FLORES_CODES = set()
SCRIPT_TO_TOKENIZER_MAPPING = {}
GLOTTOCODES = {}
def save_flores():
    ds_full = load_dataset("openlanguagedata/flores_plus", split="dev")

    all_flores = defaultdict(list)
    for lang, script, text in zip(ds_full["iso_639_3"], ds_full['iso_15924'], ds_full["text"]):
        code = f"{lang}_{script}"
        all_flores[code].append(text)
    
    for code, texts in tqdm(all_flores.items(), desc="Writing Flores files"):
        with open(f"data/flores/{code}.txt", "w", encoding="utf-8") as f:
            for text in texts:
                f.write(text + "\n")
    
def get_script_mapping():
    ds_full = load_dataset("openlanguagedata/flores_plus", split="dev")
    
    for lang, script, glottocode in zip(ds_full["iso_639_3"], ds_full['iso_15924'], ds_full["glottocode"]):
        code = f"{lang}_{script}"
        ALL_FLORES_CODES.add(code)
        GLOTTOCODES[code] = glottocode

    chinese = ["Hans", "Hant"]
    japanese = ["Jpan"]
    korean = ["Hang"]

    for code in ALL_FLORES_CODES:
        script = code.split("_")[1]
        if script in chinese:
            SCRIPT_TO_TOKENIZER_MAPPING[code] = "chinese"
        elif script in japanese:
            SCRIPT_TO_TOKENIZER_MAPPING[code] = "japanese"
        elif script in korean:
            SCRIPT_TO_TOKENIZER_MAPPING[code] = "korean"
        else:
            SCRIPT_TO_TOKENIZER_MAPPING[code] = "default"
def run():
    for code in tqdm(ALL_FLORES_CODES):
        subprocess.run(f"""uv run word-flexibility text data/flores/{code}.txt \
                    --apply-char-splitter {SCRIPT_TO_TOKENIZER_MAPPING[code]} \
                    --format lines \
                    --output results/{code}.json""", shell=True)
        
        subprocess.run(f"""uv run word-flexibility text data/flores/{code}.txt \
                    --apply-splitter flores-spm \
                    --format lines \
                    --output results/flores_spm_tokenized/{code}.json""", shell=True)
        
def compile_results():
    db = Database.load()
    rows = []
    for code in tqdm(ALL_FLORES_CODES):
        word_result = json.load(open(f"results/flores/{code}.json", "r", encoding="utf-8"))
        tokenized_result = json.load(open(f"results/flores_spm_tokenized/{code}.json", "r", encoding="utf-8"))
        language = db.get(GLOTTOCODES[code], IdType.GLOTTOCODE)
        rows.append({
            "code": code,
            "language": language.name,
            "sentences": word_result["sentences"],
            "word_flexibility": word_result["flexibility"],
            "word_lambda": word_result["lambda"],
            "word_min_pair_count": word_result["min_pair_count"],
            "word_qualifying_pairs": word_result["qualifying_pairs"],
            "word_tokens": tokenized_result["tokens"],
            "word_types": tokenized_result["types"],
            "word_ttr": tokenized_result["ttr"],
            "token_flexibility": tokenized_result["flexibility"],
            "token_lambda": tokenized_result["lambda"],
            "token_min_pair_count": tokenized_result["min_pair_count"],
            "token_qualifying_pairs": tokenized_result["qualifying_pairs"],
            "token_tokens": tokenized_result["tokens"],
            "token_types": tokenized_result["types"],
            "token_ttr": tokenized_result["ttr"],
        })

    df = pd.DataFrame(rows)
    df.to_csv("results/flores_results.csv", index=False, encoding="utf-8")
    
if __name__ == "__main__":
    # save_flores()
    # get_script_mapping()
    # run()
    # compile_results()
    
    


