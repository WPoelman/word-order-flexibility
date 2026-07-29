from random import random

from datasets import load_dataset
from collections import defaultdict
import subprocess
from scipy import stats
from tqdm import tqdm
import pandas as pd
from glob import glob
from pathlib import Path
import json
import importlib.resources as pkg_resources
from texieve import MonolingualLoader
from texieve.data import resources
from word_order_flexibility.plotting import _set_style, _regression, residualize, _format_p, _save
from word_order_flexibility.languages import UD_MAPPING
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import logging
logging.getLogger("matplotlib.texmanager").setLevel(logging.ERROR)

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


def get_ud_codes():
    df_ud = pd.read_csv("data/results-ud-full-1000-sentences-annotated.csv")
    ud_codes = zip(*[df_ud["code"].tolist(), df_ud["script"].tolist(), df_ud["Language"].tolist()])
    ud = []
    ud_langs = {}
    for code, script, language in ud_codes:
        ud_langs[f"{code}_{script}"] = language
    return ud_langs

def save_sampled_bible():
    taxi_files = glob("data/taxi-1500-pre-train/*.txt")
    for file in tqdm(taxi_files):
        code = file.split("/")[-1].split(".")[0]
        sentences = open(file, "r", encoding="utf-8").readlines()
        if len(sentences) > 10000:
            sampled_sentences = random.sample(sentences, 7958)
            with open(f"data/taxi-1500-pre-train/{code}-sampled.txt", "w", encoding="utf-8") as f:
                f.writelines(sampled_sentences)

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

                
def get_script_mapping(code):
    chinese = ["Hans", "Hant", "Hani"]
    japanese = ["Jpan", "Hira", "Kana"]
    korean = ["Hang"]
    
    script = code.split("_")[1].split("-")[0]
    if script in chinese:
        return "chinese"
    elif script in japanese:
        return "japanese"
    elif script in korean:
        return "korean"
    else:
        return "default"

            
def run():
    flores = [Path(f).stem for f in glob("data/flores/*.txt")]
    bible = [Path(f).stem for f in glob("data/taxi-1500-pre-train/*.txt")]
    wikipedia = [Path(f).stem for f in glob("data/wikipedia/*.txt")]
    
    for code in tqdm(wikipedia):
        subprocess.run(f"""uv run word-flexibility text data/wikipedia/{code}.txt \
                        --format text \
                        --apply-splitter {get_script_mapping(code)} \
                        --output results/wikipedia/{code}.json""",
                    shell=True,
                    check=True,
                )
    for code in tqdm(flores):
        subprocess.run(f"""uv run word-flexibility text data/flores/{code}.txt \
                        --apply-splitter {get_script_mapping(code)} \
                        --format lines \
                        --output results/flores/{code}.json""", 
                    shell=True,
                    check=True,
                )
    for code in tqdm(bible):
        subprocess.run(f"""uv run word-flexibility text data/taxi-1500-pre-train/{code}.txt \
                        --format tokenized \
                        --output results/bible/{code}.json""",
                    shell=True,
                    check=True,
                )

def compile_results():
    flores_rows = []
    bible_rows = []
    wikipedia_rows = []
    flores_codes = [Path(f).stem for f in glob("results/flores/*.json")]
    bible_codes = [Path(f).stem.split("-")[0] for f in glob("results/bible/*.json")]
    wikipedia_codes = [Path(f).stem for f in glob("results/wikipedia/*.json")]
    for code in tqdm(flores_codes):
        if "jpn" in code:
            continue
        result = json.load(open(f"results/flores/{code}.json", "r", encoding="utf-8"))
        flores_rows.append({
            "code": code,
            **result
        })
    for code in tqdm(bible_codes):
        if "jpn" in code:
            continue
        if code == "lzh_Hant": # this one makes the plot all weird, has WPA close to 0.
            continue
        result = json.load(open(f"results/bible/{code}.json", "r", encoding="utf-8"))
        if result["sentences"] > 10000:
            result = json.load(open(f"results/bible/{code}-sampled.json", "r", encoding="utf-8"))
        if code not in [row["code"] for row in bible_rows]:
            bible_rows.append({
                "code": code,
                **result
            })
    for code in tqdm(wikipedia_codes):
        if "jpn" in code:
            continue
        result = json.load(open(f"results/wikipedia/{code}.json", "r", encoding="utf-8"))
        if code == "yue_Hani":
            code = "yue_Hant" # to match with the UD codes
        if code  == "zho_Hani":
            code = "zho_Hans"
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
    
def _validation_panel(ax, all_results: pd.DataFrame, ud_results: pd.DataFrame, title: str = None) -> None:
    data = all_results[["code", "flexibility", "ttr"]].dropna().copy()
    data["x"] = data.index
    data["y"] = data["flexibility"]
    ud_languages = list(UD_MAPPING.keys())
    selected = all_results[all_results["code"].isin(ud_languages)][
        ["code", "flexibility", "ttr"]
    ].dropna().copy()
    ud_map_reverse = {v: k for k, v in UD_MAPPING.items()}
    ud_results["code"] = ud_results["Language"].map(ud_map_reverse)
    selected = selected.merge(
        ud_results[["code", FLEX, TTR]],
        on="code",
        how="left",
        suffixes=("", "_ud"),
    ).dropna()
    selected["x"] = selected[FLEX]
    selected["y"] = selected["flexibility"]

    other = data[~data["code"].isin(ud_languages)]
    
    ax_rank = ax          
    ax_flex = ax.twiny()
    ax_rank.xaxis.set_visible(False)
    ax_flex.xaxis.set_ticks_position("bottom")
    ax_flex.xaxis.set_label_position("bottom")
    ax_rank.scatter(
        other["x"],
        other["y"],
        s=58,
        color=MID_GRAY,
        alpha=0.42,
        edgecolor="white",
        linewidth=0.35,
        label=rf"Other languages ($n={len(other)}$)",
        zorder=1,
    )
    ax_flex.scatter(
        selected["x"],
        selected["y"],
        s=72,
        color=RED,
        alpha=0.78,
        edgecolor="white",
        linewidth=0.45,
        label=rf"UD languages ($n={len(selected)}$)",
        zorder=2,
    )
    
    ax_rank.set_ylim(data["y"].min(), data["y"].max())
    ax_flex.set_ylim(data["flexibility"].min(), data["flexibility"].max())

    ax_rank.set_xticks([])
    ax_flex.set_ylabel("Flexibility")
    

    slope, intercept = np.polyfit(selected["x"], selected["y"], 1)
    grid = np.linspace(selected["x"].min(), selected["x"].max(), 100)
    linewidth = 3.4 
    zorder = 3 
    ax_flex.plot(
        grid,
        slope * grid + intercept,
        color=RED,
        linewidth=linewidth,
        label=rf"UD fit ($n={len(selected)}$)",
        zorder=zorder,
    )

    ud_r = stats.spearmanr(selected["x"], selected["y"])
    ax_flex.text(
        0.96,
        0.05,
        f"$n={len(selected)}$, $r={ud_r.statistic:.3f}$, {_format_p(ud_r.pvalue)}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        multialignment="left",
        fontsize=ANNOTATION_SIZE,
        linespacing=1.25,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": LIGHT_GRAY,
            "edgecolor": "0.70",
            "alpha": 0.94,
        },
    )

    ax_flex.set_title(title, fontsize=AXIS_LABEL_SIZE)
    ax_flex.set_xlabel("WPA on UD", fontsize=AXIS_LABEL_SIZE)
    ax_flex.set_ylabel("WPA on " + title, fontsize=AXIS_LABEL_SIZE)
    ax_rank.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax_flex.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax_flex.grid(True, alpha=0.3)
    
def plot(output: Path = Path("results")) -> None:
    _set_style()

    flores_df = pd.read_csv("data/all-flores-results.csv")
    bible_df = pd.read_csv("data/all-bible-results.csv")
    wikipedia_df = pd.read_csv("data/all-wikipedia-results.csv")
    ud_results = pd.read_csv("results/all-1000/results.csv")
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.8), sharey=False)
    for ax, target, title in zip(axes, (flores_df, bible_df, wikipedia_df), ("Flores", "Bible", "Wikipedia"), strict=True):
        _validation_panel(ax, target, ud_results, title=title)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, output / "paper/ud_versus_all")

    
if __name__ == "__main__":
    # save_sampled_bible()
    # save_flores()
    # save_wikipedias()
    # run()
    # compile_results()
    # plot()