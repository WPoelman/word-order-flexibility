from datasets import load_dataset
from collections import defaultdict
import subprocess
from tqdm import tqdm
import pandas as pd
import json
from qq import Database, IdType 

# from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

from word_order_flexibility.languages import FLORES, UD_CODES

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
    chinese = ["Hans", "Hant"]
    japanese = ["Jpan"]
    korean = ["Hang"]

    for code in FLORES.keys():
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
    for code in tqdm(FLORES.keys()):
        subprocess.run(f"""uv run word-flexibility text data/flores/{code}.txt \
                    --apply-char-splitter {SCRIPT_TO_TOKENIZER_MAPPING[code]} \
                    --format lines \
                    --output results/{code}.json""", shell=True)
        
        subprocess.run(f"""uv run word-flexibility text data/flores/{code}.txt \
                    --apply-splitter flores-spm \
                    --format lines \
                    --output results/flores_spm_tokenized/{code}.json""", shell=True)
        
def compile_results():
    df_ud = pd.read_csv("data/results-ud-full-1000-sentences.csv")
    ud_languages = list(df_ud["Language"])
    db = Database.load()
    rows = []
    for code in tqdm(FLORES.keys()):
        word_result = json.load(open(f"results/flores/{code}.json", "r", encoding="utf-8"))
        tokenized_result = json.load(open(f"results/flores_spm_tokenized/{code}.json", "r", encoding="utf-8"))
        language = db.get(code.split("_")[0], IdType.ISO_639_3)
        name = language.name
        script = code.split("_")[-1]
        if name in ud_languages:
            ud_result = df_ud[df_ud["Language"] == name]["F_lambda_050"].values[0]
            ud_name = df_ud[df_ud["Language"] == name]["Language"].values[0]
        elif name + " (" + script + ")" in UD_MAPPING.keys():
            ud_result = df_ud[df_ud["Language"] == UD_MAPPING[name + " (" + script + ")"]]["F_lambda_050"].values[0]
            ud_name = UD_MAPPING[name + " (" + script + ")"]
        else:
            ud_result = None
            ud_name = "not_in_ud"
        speaker_count = language.speaker_count if language.speaker_count else 0
        family = language.root_family.name if language.root_family else "Unknown"
        rows.append({
            "code": code,
            "language": name + f" ({script})",
            "speaker_count": speaker_count,
            "family": family,
            "sentences": word_result["sentences"],
            "ud_result": ud_result,
            "ud_name": ud_name,
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

def compile_results():
    df_ud = pd.read_csv("data/results-ud-full-1000-sentences-annotated.csv")
    db = Database.load()
    rows = []
    ud_codes = zip(*[df_ud["code"].tolist(), df_ud["script"].tolist(), 
                     df_ud["run_again"].tolist(), df_ud["Language"].tolist(),
                     df_ud["F_lambda_050"].tolist()])
    # for code in tqdm(flores_codes):
    #     word_result = json.load(open(f"results/bible/{code}.json", "r", encoding="utf-8"))
    #     if word_result["sentences"] > 10000:
    #         continue
    #     language = db.get(code.split("_")[0], IdType.ISO_639_3)
    #     name = language.name
    #     script = code.split("_")[-1].split("-")[0]
    #     if name in ud_languages:
    #         ud_result = df_ud[df_ud["Language"] == name]["F_lambda_050"].values[0]
    #         ud_name = df_ud[df_ud["Language"] == name]["Language"].values[0]
    #     elif name + " (" + script + ")" in UD_CODES.keys():
    #         ud_result = df_ud[df_ud["Language"] == UD_CODES[name + " (" + script + ")"]]["F_lambda_050"].values[0]
    #         ud_name = UD_CODES[name + " (" + script + ")"]
    #     else:
    #         ud_result = None
    #         ud_name = "not_in_ud"
    #     speaker_count = language.speaker_count if language.speaker_count else 0
    #     family = language.root_family.name if language.root_family else "Unknown"
    for code, script, run_again, language, result in tqdm(ud_codes):
        # if run_again == "yes":
        #     continue
        try:
            flores_result = json.load(open(f"results/flores/{code}_{script}.json", "r", encoding="utf-8"))
        except FileNotFoundError:
            print(f"File not found for {code}_{script}.json")
            continue
        rows.append({
            "code": code,
            "language": language,
            "sentences": flores_result["sentences"],
            "ud_result": result,
            "word_flexibility": flores_result["flexibility"],
            "word_lambda": flores_result["lambda"],
            "word_min_pair_count": flores_result["min_pair_count"],
            "word_qualifying_pairs": flores_result["qualifying_pairs"],
        })
        
    df = pd.DataFrame(rows)
    df.to_csv("results/flores_results-1.csv", index=False, encoding="utf-8")
    
def correlation(frame: pd.DataFrame, x: str, y: str, controlled: bool) -> dict:
    data = frame[[x, y, TTR]].replace([np.inf, -np.inf], np.nan).dropna()
    xv, yv = data[x].to_numpy(), data[y].to_numpy()
    # if controlled:
    #     xv, yv = residualize(data[x], data[TTR]), residualize(data[y], data[TTR])
    pearson = stats.pearsonr(xv, yv)
    spearman = stats.spearmanr(xv, yv)
    return {
        "x": x,
        "y": y,
        "control": "TTR" if controlled else "none",
        "n": len(data),
        "pearson_r": pearson.statistic,
        "pearson_p": pearson.pvalue,
        "spearman_rho": spearman.statistic,
        "spearman_p": spearman.pvalue,
    }
    
def _regression(ax, x, y, color, label) -> None:
    ax.scatter(x, y, color=color, alpha=0.75, s=60, edgecolor="white", linewidth=0.4)
    slope, intercept = np.polyfit(x, y, 1)
    grid = np.linspace(min(x), max(x), 100)
    ax.plot(grid, slope * grid + intercept, color=color, linewidth=2.3, label=label)

    
def plot():
    df = pd.read_csv("results/flores_results-1.csv")
    df = df.sort_values('word_flexibility', ascending=False)
    n = len(df)
    fig, ax = plt.subplots(figsize=(20, 5))
    ax1 = ax.twinx()

    sns.scatterplot(
        y="word_flexibility", x="code", data=df, color=GREEN, alpha=0.75, s=40, ax=ax
    )
    sns.scatterplot(
        y="ud_result", x="code", data=df,
        color=RED, alpha=0.75, s=40, ax=ax1,
    )
    
    flores, ud = [], []
    for i, j, k in zip(df["word_flexibility"], df["ud_result"], df["language"]):
        if k != "not_in_ud":
            flores.append(i)
            ud.append(j)
    # _regression(ax1, flores, ud, color=SLATE, label="Regression line")
    rho, pval = stats.spearmanr(flores, ud)
    ax1.annotate(f"Spearman's rho: {rho:.3f}\np-value: {pval:.3f}", xy=(0.02, 0.02), xycoords="axes fraction", fontsize=ANNOTATION_SIZE, color=SLATE)
    # ax1.set_ylabel("Token TTR")
    ax.set_ylabel("WPA")
    ax1.set_ylabel("WPA")
    ax.set_xlabel("Language")

    # label every 5th tick with the language name (not the code)
    # languages = df["language"].tolist()
    # labels = [lang.split(' (')[0].split()[-1].split("-")[-1] if i % 3 == 0 else "" for i, lang in enumerate(languages)]
    labels = [lang if lang != "not_in_ud" else "" for i, lang in enumerate(df["language"].tolist())]
    # labels = [lang if speaker_count > 10000000 else "" for lang, speaker_count in zip(df["language"], df["speaker_count"])]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    # ax1.legend(["Token TTR"], fontsize=12, loc="upper right")
    # ax.legend(["Word Flexibility"], fontsize=12, loc="upper left")
    ax.legend(["WPA (Flores)"], fontsize=12, loc="upper left")
    ax1.legend(["WPA (UD)"], fontsize=12, loc="upper right")
    ax.margins(y=0.002)
    plt.tight_layout()
    plt.savefig("flores_plot-1.png", dpi=200, bbox_inches="tight")
    plt.show()
    
if __name__ == "__main__":
    # save_flores()
    # get_script_mapping()
    # run()
    compile_results()
    plot()
    
    


