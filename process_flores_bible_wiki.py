from datasets import load_dataset
from collections import defaultdict
import subprocess
from tqdm import tqdm
import pandas as pd
from glob import glob
from pathlib import Path
import json
import importlib.resources as pkg_resources
from texieve import MonolingualLoader
from texieve.data import resources
from word_order_flexibility.plotting import _set_style, _regression, residualize
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

GREEN = "#1B7837"
RED = "#8E1B1B"
SLATE = "#4C566A"
MID_GRAY = "#9A9A9A"
LIGHT_GRAY = "#F2F2F2"
AXIS_LABEL_SIZE = 22
TICK_LABEL_SIZE = 15
ANNOTATION_SIZE = 18
LEGEND_SIZE = 18
DECAYS = (0.25, 0.5, 0.75, 1.0)
SIZES = (1000, 2000, 5000, 10000)
FLEX = "F_lambda_050"
TTR = "type_token_ratio"

SCRIPT_TO_TOKENIZER_MAPPING = {}

def get_ud_codes():
    df_ud = pd.read_csv("data/results-ud-full-1000-sentences-annotated.csv")
    ud_codes = zip(*[df_ud["code"].tolist(), df_ud["script"].tolist(), df_ud["Language"].tolist()])
    ud = []
    ud_langs = {}
    for code, script, language in ud_codes:
        ud_langs[f"{code}_{script}"] = language
    return ud_langs

def get_all_wikipedias():
    with pkg_resources.open_text(resources, "wiki_mappings.json") as file:
        lang_mappings = json.load(file)
    iso_script_codes = []
    for wiki_code, mapping in lang_mappings.items():
        iso = mapping["alpha3"]
        scripts = mapping["scripts"]
        for script in scripts:
            iso_script_codes.append(f"{iso}_{script}")
    return lang_mappings, iso_script_codes

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
                
def save_wikipedias():
    wikipedias, _ = get_all_wikipedias()
    for wiki_code in tqdm(wikipedias.keys(), desc="Processing Wikipedia datasets"):
        iso = wikipedias[wiki_code]["alpha3"]
        scripts = wikipedias[wiki_code]["scripts"]
        for script in scripts:
            code = f"{iso}_{script}"
            if code in [Path(f).stem for f in glob("data/wikipedia/*.txt")]:
                continue
        try:
            data = load_dataset("WikiQuality/pre_filtered", wiki_code, streaming=True, split="train").shuffle()
        except:
            try:
                loader = MonolingualLoader(iso)
                loader.load(sources=["wiki"], streaming=True)
        #           loader.pre_filter(script_regex=True)
        #          loader.deduplicate(exact_match=True, min_hash=True)
                data = loader.data['train'].shuffle()
            except:
                print(f"Failed to load {iso} ({wiki_code})")
                continue
        data = data.take(1000)
        data = [i['text'] for i in data]
        for script in scripts:
            code = f"{iso}_{script}"
            with open(f'data/wikipedia/{code}.txt', "w", encoding="utf-8") as outfile:
                outfile.writelines(data)

                
def get_script_mapping():
    chinese = ["Hans", "Hant", "Hani"]
    japanese = ["Jpan", "Hira", "Kana"]
    korean = ["Hang"]

    for code in get_ud_codes():
        script = code.split("_")[1]
        if script in chinese:
            SCRIPT_TO_TOKENIZER_MAPPING[code] = "chinese"
        elif script in japanese:
            SCRIPT_TO_TOKENIZER_MAPPING[code] = "japanese"
        elif script in korean:
            SCRIPT_TO_TOKENIZER_MAPPING[code] = "korean"
        else:
            SCRIPT_TO_TOKENIZER_MAPPING[code] = "default"
            
def process_code(code: str):
    # if code not in [Path(f).stem for f in glob("results/flores/*.json")]:
    #     subprocess.run(f"""uv run word-flexibility text data/flores/{code}.txt \
    #             --apply-splitter {SCRIPT_TO_TOKENIZER_MAPPING[code]} \
    #             --format lines \
    #             --output results/flores/{code}.json""", 
    #         shell=True,
    #         check=True,
    #     )
    # if code not in [Path(f).stem for f in glob("results/bible/*.json")]:
    #     subprocess.run(f"""uv run word-flexibility text data/taxi-1500-pre-train/{code}.txt \
    #             --format tokenized \
    #             --output results/bible/{code}.json""",
    #         shell=True,
    #         check=True,
    #     )
    if code not in [Path(f).stem for f in glob("results/wikipedia/*.json")]:
        subprocess.run(f"""uv run word-flexibility text data/wikipedia/{code}.txt \
                --format text \
                --apply-splitter {SCRIPT_TO_TOKENIZER_MAPPING[code]} \
                --output results/wikipedia/{code}.json""",
            shell=True,
            check=True,
        )
    return code
            
def run():
    get_script_mapping()
    _, wikipedias = get_all_wikipedias()
    
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_code, code): code for code in wikipedias}
        for future in tqdm(as_completed(futures), total=len(futures)):
            code = futures[future]
            try:
                future.result()
            except subprocess.CalledProcessError as e:
                print(f"Failed on {code}: {e}")

def compile_results():
    flores_rows = []
    bible_rows = []
    wikipedia_rows = []
    flores_codes = [Path(f).stem for f in glob("results/flores/*.json")]
    bible_codes = [Path(f).stem for f in glob("results/bible/*.json")]
    wikipedia_codes = [Path(f).stem for f in glob("results/wikipedia/*.json")]
    for code in tqdm(flores_codes):
        result = json.load(open(f"results/flores/{code}.json", "r", encoding="utf-8"))
        flores_rows.append({
            "code": code,
            **result
        })
    for code in tqdm(bible_codes):
        result = json.load(open(f"results/bible/{code}.json", "r", encoding="utf-8"))
        bible_rows.append({
            "code": code,
            **result
        })
    for code in tqdm(wikipedia_codes):
        result = json.load(open(f"results/wikipedia/{code}.json", "r", encoding="utf-8"))
        wikipedia_rows.append({
            "code": code,
            **result
        })
        
    flores_df = pd.DataFrame(flores_rows)
    flores_df.to_csv("data/all-flores-results.csv", index=False, encoding="utf-8")
    bible_df = pd.DataFrame(bible_rows)
    bible_df.to_csv("data/all-bible-results.csv", index=False, encoding="utf-8")
    wikipedia_df = pd.DataFrame(wikipedia_rows)
    wikipedia_df.to_csv("data/all-wikipedia-results.csv", index=False, encoding="utf-8")

    
if __name__ == "__main__":
    # save_flores()
    save_wikipedias()
    run()
    compile_results()
    # plot()
