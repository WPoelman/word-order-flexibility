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
    with pkg_resources.open_text(resources, "language_mappings.json") as file:
        lang_mappings = json.load(file)
        
    return lang_mappings

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
    wikipedias = get_all_wikipedias()
    ud_codes = get_ud_codes()
    for code in tqdm(ud_codes, desc="Processing Wikipedia datasets"):
        iso = code.split("_")[0]
        if code in [Path(f).stem for f in glob("data/wikipedia/*.txt")]:
            continue
        script = code.split("_")[1]
        if script in wikipedias.get(iso, {}).get("scripts", []):
            wiki_code = wikipedias[iso]["bcp_47_code"]
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
            with open(f'data/wikipedia/{code}.txt', "w", encoding="utf-8") as outfile:
                outfile.writelines(data)

                
def get_script_mapping():
    chinese = ["Hans", "Hant"]
    japanese = ["Jpan"]
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
    if code in [Path(f).stem for f in glob("data/flores/*.txt")]:
        subprocess.run(f"""uv run word-flexibility text data/flores/{code}.txt \
                --apply-splitter {SCRIPT_TO_TOKENIZER_MAPPING[code]} \
                --format lines \
                --output results/flores/{code}.json""", 
            shell=True,
            check=True,
        )
    if code in [Path(f).stem for f in glob("data/taxi-1500-pre-train/*.txt")]: 
        subprocess.run(f"""uv run word-flexibility text data/taxi-1500-pre-train/{code}.txt \
                --format tokenized \
                --output results/bible/{code}.json""",
            shell=True,
            check=True,
        )
    if code in [Path(f).stem for f in glob("data/wikipedia/*.txt")]:
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
    ud_codes = get_ud_codes()
    
    max_workers = 7  # tune based on CPU cores / memory

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_code, code): code for code in ud_codes}
        for future in tqdm(as_completed(futures), total=len(futures)):
            code = futures[future]
            try:
                future.result()
            except subprocess.CalledProcessError as e:
                print(f"Failed on {code}: {e}")

def compile_results():
    rows = []
    ud_codes = get_ud_codes()
    df_ud = pd.read_csv("data/results-ud-full-1000-sentences-annotated.csv")
    for code in tqdm(ud_codes):
        try:
            flores_result = json.load(open(f"results/flores/{code}.json", "r", encoding="utf-8"))
        except:
            flores_result = {"sentences": 0.5, "flexibility": 0.5, "lambda": 0.5, "min_pair_count": 0.5, "qualifying_pairs": 0.5, "ttr": 0.5}
        try:
            bible_result = json.load(open(f"results/bible/{code}.json", "r", encoding="utf-8"))
        except:
            bible_result = {"sentences": 0.5, "flexibility": 0.5, "lambda": 0.5, "min_pair_count": 0.5, "qualifying_pairs": 0.5, "ttr": 0.5}
        try:
            wikipedia_result = json.load(open(f"results/wikipedia/{code}.json", "r", encoding="utf-8"))
        except:
            wikipedia_result = {"sentences": 0.5, "flexibility": 0.5, "lambda": 0.5, "min_pair_count": 0.5, "qualifying_pairs": 0.5, "ttr": 0.5}
        rows.append({
            "code": code,
            "language": ud_codes[code],
            "flores_sentences": flores_result["sentences"],
            "bible_sentences": bible_result["sentences"],
            "wikipedia_sentences": wikipedia_result["sentences"],
            "ud_sentences": df_ud[df_ud["code"] == code.split("_")[0]]["n_sentences"].values[0],
            "ud_flexibility": df_ud[df_ud["code"] == code.split("_")[0]]["F_lambda_050"].values[0],
            "flores_flexibility": flores_result["flexibility"],
            "bible_flexibility": bible_result["flexibility"],
            "wikipedia_flexibility": wikipedia_result["flexibility"],
            "flores_min_pair_count": flores_result["min_pair_count"],
            "bible_min_pair_count": bible_result["min_pair_count"],
            "wikipedia_min_pair_count": wikipedia_result["min_pair_count"],
            "flores_qualifying_pairs": flores_result["qualifying_pairs"],
            "bible_qualifying_pairs": bible_result["qualifying_pairs"],
            "wikipedia_qualifying_pairs": wikipedia_result["qualifying_pairs"],
            "flores_ttr": flores_result["ttr"],
            "bible_ttr": bible_result["ttr"],
            "wikipedia_ttr": wikipedia_result["ttr"],
        })
        
    df = pd.DataFrame(rows)
    df.to_csv("data/combined_results.csv", index=False, encoding="utf-8")
    
def validation_plot(df: pd.DataFrame, output: str | Path, *, control_ttr: bool = True) -> None:
    """Plot flexibility against HDE and ROE for the public CSV interface."""
    _set_style()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for ax, target, color in zip(axes, ("bible", "wikipedia", "flores"), (RED, GREEN, SLATE), strict=True):
        target_flexibility = f"{target}_flexibility"
        data = df[["ud_flexibility", target_flexibility, f"{target}_ttr"]].astype(float).dropna()
        x, y = data[target_flexibility].to_numpy(), data["ud_flexibility"].to_numpy()
        if control_ttr:
            x = residualize(x, data[f"{target}_ttr"])
            y = residualize(y, data[f"{target}_ttr"])
        _regression(ax, x, y, color, "Linear fit")
        suffix = " residual" if control_ttr else ""
        ax.set_xlabel("WPA on " + target + suffix)
        ax.set_ylabel("WPA on UD" + suffix)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    
if __name__ == "__main__":
    # save_flores()
    # save_wikipedias()
    run()
    compile_results()

    df = pd.read_csv("data/combined_results.csv")
    validation_plot(df, "data/validation_plot.png", control_ttr=False)