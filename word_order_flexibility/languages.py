# Original Futrell languages (Bengali dropped: only 56 sentences in UD 2.17)
FUTRELL_LANGUAGES = [
    ("Ancient Greek", "UD_Ancient_Greek-Perseus"),
    ("Arabic", "UD_Arabic-PADT"),
    ("Basque", "UD_Basque-BDT"),
    ("Bulgarian", "UD_Bulgarian-BTB"),
    ("Catalan", "UD_Catalan-AnCora"),
    ("Croatian", "UD_Croatian-SET"),
    ("Czech", "UD_Czech-CLTT"),
    ("Danish", "UD_Danish-DDT"),
    ("Dutch", "UD_Dutch-LassySmall"),
    ("English", "UD_English-EWT"),
    ("Estonian", "UD_Estonian-EWT"),
    ("Finnish", "UD_Finnish-FTB"),
    ("French", "UD_French-ALTS"),
    ("German", "UD_German-GSD"),
    ("Hebrew", "UD_Hebrew-HTB"),
    ("Hindi", "UD_Hindi-HDTB"),
    ("Hungarian", "UD_Hungarian-Szeged"),
    ("Irish", "UD_Irish-IDT"),
    ("Italian", "UD_Italian-ISDT"),
    ("Japanese", "UD_Japanese-GSD"),
    ("Latin", "UD_Latin-CIRCSE"),
    ("Modern Greek", "UD_Greek-GDT"),
    ("Persian", "UD_Persian-PerDT"),
    ("Portuguese", "UD_Portuguese-CINTIL"),
    ("Romanian", "UD_Romanian-RRT"),
    ("Russian", "UD_Russian-GSD"),
    ("Slovak", "UD_Slovak-SNK"),
    ("Slovenian", "UD_Slovenian-SST"),
    ("Spanish", "UD_Spanish-GSD"),
    ("Swedish", "UD_Swedish-LinES"),
    ("Tamil", ["UD_Tamil-TTB", "UD_Tamil-MWTT"]),
    ("Telugu", "UD_Telugu-MTG"),
    ("Turkish", "UD_Turkish-IMST"),
]

# Languages with >= 1000 sentences (combining all treebanks)
# Some historic languages or treebanks with issues are commented out

LANGUAGES = {
    "Abkhaz": [
        "UD_Abkhaz-AbNC",
    ],
    "Afrikaans": [
        "UD_Afrikaans-AfriBooms",
    ],
    "Akkadian": [
        "UD_Akkadian-PISANDUB",
        "UD_Akkadian-RIAO",
    ],
    "Alemannic": [
        "UD_Alemannic-UZH",
        "UD_Alemannic-DIVITAL",
    ],
    "Amharic": [
        "UD_Amharic-ATT",
    ],
    "Ancient Greek": [
        "UD_Ancient_Greek-Perseus",
        "UD_Ancient_Greek-PROIEL",
        "UD_Ancient_Greek-PTNK",
    ],
    "Ancient Hebrew": [
        "UD_Ancient_Hebrew-PTNK",
    ],
    "Arabic": [
        # "UD_Arabic-NYUAD", # is only underscores? could be copyright according to Miryam...
        "UD_Arabic-PUD",
        "UD_Arabic-PADT",
    ],
    "Eastern Armenian": [
        "UD_Armenian-BSUT",
        "UD_Armenian-ArmTDP",
    ],
    "Bambara": [
        "UD_Bambara-CRB",
    ],
    "Basque": [
        "UD_Basque-BDT",
    ],
    # "Bavarian": [
    #     "UD_Bavarian-MaiBaam",
    # ],
    "Belarusian": [
        "UD_Belarusian-HSE",
    ],
    "Bororo": [
        "UD_Bororo-BDT",
    ],
    "Bulgarian": [
        "UD_Bulgarian-BTB",
    ],
    "Cantonese": [
        "UD_Cantonese-HK",
    ],
    "Catalan": [
        "UD_Catalan-AnCora",
    ],
    "Chinese Simplified": [ 
        "UD_Chinese-GSDSimp",
        "UD_Chinese-CFL",
        "UD_Chinese-PatentChar",
        "UD_Chinese-Beginner",
    ],
    "Chinese Traditional": [
        "UD_Chinese-GSD",
        "UD_Chinese-HK",
        "UD_Chinese-PUD",
    ],
    "Chintang": [
        "UD_Chintang-CTNTB",
    ],
    "Chukchi": [
        "UD_Chukchi-HSE",
    ],
    # "Classical Armenian": [
    #     "UD_Classical_Armenian-CAVaL",
    # ],
    # "Classical Chinese": [
    #     "UD_Classical_Chinese-Kyoto",
    #     "UD_Classical_Chinese-TueCL",
    # ],
    "Coptic": [
        "UD_Coptic-Scriptorium",
        "UD_Coptic-Bohairic",
    ],
    "Croatian": [
        "UD_Croatian-SET",
    ],
    "Czech": [
        "UD_Czech-PDTC",
        "UD_Czech-CLTT",
        "UD_Czech-CAC",
        "UD_Czech-PUD",
        "UD_Czech-FicTree",
        # "UD_Czech-Poetry",
    ],
    "Danish": [
        "UD_Danish-DDT",
    ],
    "Dutch": [
        "UD_Dutch-LassySmall",
        "UD_Dutch-Alpino",
    ],
    "Egyptian": [
        "UD_Egyptian-UJaen",
    ],
    "English": [
        "UD_English-EWT",
        "UD_English-LinES",
        "UD_English-GUM",
        # "UD_English-ESLSpok", # spoken
        "UD_English-GENTLE",
        # "UD_English-GUMReddit",  # redacted (all forms are underscores)
        "UD_English-ParTUT",
        "UD_English-CTeTex",
        "UD_English-PUD",
        # "UD_English-CHILDES", # child-directed and child-produced spoken
        "UD_English-LittlePrince",
        "UD_English-Atis",
        "UD_English-Pronouns",
    ],
    "Erzya": [
        "UD_Erzya-JR",
    ],
    "Estonian": [
        "UD_Estonian-EWT",
        "UD_Estonian-EDT",
    ],
    "Faroese": [
        # "UD_Faroese-FarPaHC", # historical
        "UD_Faroese-OFT",
    ],
    "Finnish": [
        "UD_Finnish-FTB",
        "UD_Finnish-TDT",
        "UD_Finnish-PUD",
        "UD_Finnish-OOD",
    ],
    "French": [
        # "UD_French-ParisStories", # spoken
        # "UD_French-Rhapsodie", # spoken
        "UD_French-ParTUT", 
        "UD_French-PUD",
        # "UD_French-PoitevinDIVITAL", # dialect
        "UD_French-FQB",
        "UD_French-Sequoia",
        "UD_French-GSD",
        # "UD_French-ALTS", # historical
    ],
    "Galician": [
        "UD_Galician-PUD",
        "UD_Galician-CTG",
        "UD_Galician-TreeGal",
    ],
    "Georgian": [
        "UD_Georgian-GLC",
        "UD_Georgian-GNC",
    ],
    "German": [
        "UD_German-HDT",
        # "UD_German-LIT", # historical
        "UD_German-PUD",
        "UD_German-GSD",
    ],
    "Gothic": [
        "UD_Gothic-PROIEL",
    ],
    "Modern Greek": [
        # "UD_Greek-Lesbian", # dialect
        "UD_Greek-GUD",
        "UD_Greek-GDT",
        # "UD_Greek-Messinian", # dialect
        "UD_Greek-Cretan", 
    ],
    "Guajajara": [
        "UD_Guajajara-TuDeT",
    ],
    "Haitian Creole": [
        "UD_Haitian_Creole-Autogramm",
        "UD_Haitian_Creole-Adolphe",
    ],
    "Hausa": [
        "UD_Hausa-WesternAutogramm",
        "UD_Hausa-SouthernAutogramm",
        "UD_Hausa-NorthernAutogramm",
    ],
    "Hebrew": [
        "UD_Hebrew-IAHLTknesset",
        "UD_Hebrew-IAHLTwiki",
        "UD_Hebrew-HTB",
    ],
    "Highland Puebla Nahuatl": [
        "UD_Highland_Puebla_Nahuatl-ITML",
    ],
    "Hindi": [
        "UD_Hindi-HDTB",
        "UD_Hindi-PUD",
    ],
    "Hungarian": [
        "UD_Hungarian-Szeged",
    ],
    "Icelandic": [
        "UD_Icelandic-GC",
        "UD_Icelandic-Modern",
        "UD_Icelandic-PUD",
        # "UD_Icelandic-IcePaHC", # historical
    ],
    "Indonesian": [
        "UD_Indonesian-CSUI",
        "UD_Indonesian-PUD",
        "UD_Indonesian-GSD",
    ],
    "Irish": [
        "UD_Irish-IDT",
        # "UD_Irish-Cadhan", # historical
        "UD_Irish-TwittIrish",
    ],
    "Italian": [
        "UD_Italian-ParTUT",
        "UD_Italian-VIT",
        "UD_Italian-PoSTWITA",
        "UD_Italian-ParlaMint",
        "UD_Italian-TWITTIRO",
        # "UD_Italian-Old",
        "UD_Italian-Valico",
        "UD_Italian-PUD",
        "UD_Italian-MarkIT",
        # "UD_Italian-KIParlaForest", # spoken
        "UD_Italian-ISDT",
    ],
    "Japanese": [
        "UD_Japanese-GSD",
        "UD_Japanese-PUD",
    ],
    "Javanese": [
        "UD_Javanese-CSUI",
    ],
    "Kazakh": [
        "UD_Kazakh-KTB",
    ],
    "Khoekhoe": [
        "UD_Khoekhoe-KDT",
    ],
    "Kiche": [
        "UD_Kiche-IU",
    ],
    "Korean": [
        "UD_Korean-KSL",
        "UD_Korean-LittlePrince",
        "UD_Korean-Kaist",
        "UD_Korean-GSD",
        "UD_Korean-PUD",
    ],
    "Kyrgyz": [
        "UD_Kyrgyz-KTMU",
        "UD_Kyrgyz-TueCL",
    ],
    "Latin": [
        "UD_Latin-ITTB",
        "UD_Latin-LLCT",
        "UD_Latin-CIRCSE",
        "UD_Latin-PROIEL",
        "UD_Latin-Perseus",
        "UD_Latin-UDante",
    ],
    "Latvian": [
        "UD_Latvian-LVTB",
        "UD_Latvian-Cairo",
    ],
    "Lithuanian": [
        "UD_Lithuanian-HSE",
        "UD_Lithuanian-ALKSNIS",
    ],
    # "Low Saxon": [
    #     "UD_Low_Saxon-LSDC",
    # ],
    # "Maghrebi Arabic French": [
    #     "UD_Maghrebi_Arabic_French-Arabizi",
    # ],
    "Maltese": [
        "UD_Maltese-MUDT",
    ],
    "Manx": [
        "UD_Manx-Cadhan",
    ],
    "Mbya Guarani": [
        # "UD_Mbya_Guarani-Dooley",  # redacted (98% underscores)
        "UD_Mbya_Guarani-Thomas",
    ],
    # "Middle French": [
    #     "UD_Middle_French-ALTM",
    #     "UD_Middle_French-PROFITEROLE",
    # ],
    "Naija": [
        "UD_Naija-NSC",
    ],
    "Nheengatu": [
        "UD_Nheengatu-CompLin",
    ],
    "North Sami": [
        "UD_North_Sami-Giella",
    ],
    "Norwegian Nynorsk": [ # Norwegian has 2 treebanks, with different iso3 codes
        "UD_Norwegian-Nynorsk", # nno
    ],
    "Norwegian Bokmaal": [
        "UD_Norwegian-Bokmaal", # nob
    ],
    "Occitan": [
        "UD_Occitan-TTB",
    ],
    # "Old Church Slavonic": [
    #     "UD_Old_Church_Slavonic-PROIEL",
    # ],
    # "Old East Slavic": [
    #     "UD_Old_East_Slavic-Birchbark",
    #     "UD_Old_East_Slavic-RNC",
    #     "UD_Old_East_Slavic-TOROT",
    #     "UD_Old_East_Slavic-Ruthenian",
    # ],
    # "Old French": [
    #     "UD_Old_French-PROFITEROLE",
    #     "UD_Old_French-ALTM",
    # ],
    # "Old Occitan": [
    #     "UD_Old_Occitan-CorAG",
    # ],
    "Ottoman Turkish": [
        "UD_Ottoman_Turkish-BOUN",
        "UD_Ottoman_Turkish-DUDU",
    ],
    "Persian": [
        "UD_Persian-Seraji",
        "UD_Persian-PerDT",
    ],
    "Polish": [
        "UD_Polish-PDB",
        "UD_Polish-PUD",
        "UD_Polish-LFG",
        # "UD_Polish-MPDT", # historical
    ],
    "Pomak": [
        "UD_Pomak-Philotis", # note that Pomak uses the same iso3 code as bulgarian bul
    ], # but the bulgarian UD is in Cyrl and this is in Latn script
    "Portuguese": [
        "UD_Portuguese-Porttinari",
        # "UD_Portuguese-Bosque", # both Portuguese and Brazilian
        "UD_Portuguese-PUD",
        "UD_Portuguese-CINTIL",
        "UD_Portuguese-GSD",
        # "UD_Portuguese-DANTEStocks", # brazilian portuguese
        # "UD_Portuguese-PetroGold", # brazilian brazilian portuguese
    ],
    "Romanian": [
        # "UD_Romanian-Nonstandard", # non-standard romanian
        # "UD_Romanian-ArT", # dialect
        "UD_Romanian-TueCL",
        "UD_Romanian-RRT",
        "UD_Romanian-SiMoNERo",
        # "UD_Romanian-MolDoRo", # dialect and in Cyrl, others are Latn
    ],
    "Russian": [
        "UD_Russian-PUD",
        "UD_Russian-Taiga",
        "UD_Russian-SynTagRus",
        # "UD_Russian-Poetry", # historical poetry
        "UD_Russian-GSD",
    ],
    "Sanskrit Vedic": [ # vsn
        "UD_Sanskrit-Vedic", # separated to Vedic and Classical
    ],
    "Sanskrit Classical": [ # cls
        "UD_Sanskrit-UFAL",
    ],
    "Scottish Gaelic": [
        "UD_Scottish_Gaelic-ARCOSG",
    ],
    "Serbian": [
        "UD_Serbian-SET",
    ],
    "Sindhi": [
        "UD_Sindhi-Isra",
    ],
    "Slovak": [
        "UD_Slovak-SNK",
    ],
    "Slovenian": [
        # "UD_Slovenian-SST", # spoken
        "UD_Slovenian-SSJ",
    ],
    "Spanish": [
        "UD_Spanish-GSD",
        "UD_Spanish-AnCora",
        # "UD_Spanish-COSER", # spoken
        "UD_Spanish-PUD",
    ],
    "Swedish": [
        "UD_Swedish-LinES",
        "UD_Swedish-PUD",
        "UD_Swedish-Talbanken",
        "UD_Swedish-SweLL",
        # "UD_Swedish-Old",
    ],
    "Tamil": [
        "UD_Tamil-TTB",
        "UD_Tamil-MWTT",
    ],
    "Telugu": [
        "UD_Telugu-MTG",
    ],
    "Thai": [
        "UD_Thai-TUD",
        "UD_Thai-PUD",
    ],
    "Turkish": [
        "UD_Turkish-BOUN",
        "UD_Turkish-Penn",
        "UD_Turkish-Atis",
        "UD_Turkish-FrameNet",
        "UD_Turkish-Tourism",
        "UD_Turkish-TueCL",
        "UD_Turkish-PUD",
        "UD_Turkish-GB",
        "UD_Turkish-Kenet",
        "UD_Turkish-IMST",
    ],
    "Ukrainian": [
        "UD_Ukrainian-IU",
        "UD_Ukrainian-ParlaMint",
    ],
    "Urdu": [
        "UD_Urdu-UDTB",
    ],
    "Uyghur": [
        "UD_Uyghur-UDT",
    ],
    "Uzbek": [
        "UD_Uzbek-UT",
        "UD_Uzbek-TueCL",
        "UD_Uzbek-UzUDT",
    ],
    "Vietnamese": [
        "UD_Vietnamese-TueCL",
        "UD_Vietnamese-VTB",
    ],
    "Welsh": [
        "UD_Welsh-CCG",
    ],
    "Western Armenian": [
        "UD_Western_Armenian-ArmTDP",
    ],
    "Wolof": [
        "UD_Wolof-WTB",
    ],
    "Yiddish": [
        "UD_Yiddish-YiTB",
    ],
    "Zaar": [
        "UD_Zaar-Autogramm",
    ],
}

UD_MAPPING = {
    "abk_Cyrl": "Abkhaz",
    "afr_Latn": "Afrikaans",
    "akk_Latn": "Akkadian",
    "gsw_Latn": "Alemannic",
    "amh_Ethi": "Amharic",
    "grc_Grek": "Ancient Greek",
    "hbo_Hebr": "Ancient Hebrew",
    "ara_Arab": "Arabic",
    "hye_Armn": "Armenian",
    "bam_Latn": "Bambara",
    "eus_Latn": "Basque",
    "bel_Cyrl": "Belarusian",
    "bor_Latn": "Bororo",
    "bul_Cyrl": "Bulgarian",
    "yue_Hant": "Cantonese",
    "cat_Latn": "Catalan",
    "zho_Hant": "Chinese Traditional",
    "zho_Hans": "Chinese Simplified",
    "ctn_Latn": "Chintang",
    "1000_ckt": "Chukchi",
    "cop_Copt": "Coptic",
    "hrv_Latn": "Croatian",
    "ces_Latn": "Czech",
    "dan_Latn": "Danish",
    "nld_Latn": "Dutch",
    "egy_Latn": "Egyptian",
    "eng_Latn": "English",
    "myv_Cyrl": "Erzya",
    "est_Latn": "Estonian",
    "fao_Latn": "Faroese",
    "fin_Latn": "Finnish",
    "fra_Latn": "French",
    "glg_Latn": "Galician",
    "kat_Geor": "Georgian",
    "deu_Latn": "German",
    "got_Latn": "Gothic",
    "ell_Grek": "Modern Greek",
    "gub_Latn": "Guajajara",
    "hat_Latn": "Haitian Creole",
    "hau_Latn": "Hausa",
    "heb_Hebr": "Hebrew",
    "azz_Latn": "Highland Puebla Nahuatl",
    "hin_Deva": "Hindi",
    "hun_Latn": "Hungarian",
    "isl_Latn": "Icelandic",
    "ind_Latn": "Indonesian",
    "gle_Latn": "Irish",
    "ita_Latn": "Italian",
    "jpn_Jpan": "Japanese",
    "jav_Latn": "Javanese",
    "kaz_Cyrl": "Kazakh",
    "naq_Latn": "Khoekhoe",
    "quc_Latn": "Kiche",
    "kor_Hang": "Korean",
    "kir_Cyrl": "Kyrgyz",
    "lat_Latn": "Latin",
    "lav_Latn": "Latvian",
    "lit_Latn": "Lithuanian",
    "mlt_Latn": "Maltese",
    "glv_Latn": "Manx",
    "pcm_Latn": "Naija",
    "yrl_Latn": "Nheengatu",
    "sme_Latn": "North Sami",
    "nno_Latn": "Norwegian Nynorsk",
    "nob_Latn": "Norwegian Bokmaal",
    "oci_Latn": "Occitan",
    "ota_Latn": "Ottoman Turkish",
    "fas_Arab": "Persian",
    "pol_Latn": "Polish",
    "bul_Latn": "Pomak",
    "por_Latn": "Portuguese",
    "ron_Latn": "Romanian",
    "rus_Cyrl": "Russian",
    # "cls_Latn": "Sanskrit Classical",
    "vsn_Latn": "Sanskrit Vedic",
    "gla_Latn": "Scottish Gaelic",
    "srp_Latn": "Serbian",
    "snd_Arab": "Sindhi",
    "slk_Latn": "Slovak",
    "slv_Latn": "Slovenian",
    "spa_Latn": "Spanish",
    "swe_Latn": "Swedish",
    "tam_Taml": "Tamil",
    "tel_Telu": "Telugu",
    "tha_Thai": "Thai",
    "tur_Latn": "Turkish",
    "ukr_Cyrl": "Ukrainian",
    "urd_Arab": "Urdu",
    "uig_Arab": "Uyghur",
    "uzb_Latn": "Uzbek",
    "vie_Latn": "Vietnamese",
    "cym_Latn": "Welsh",
    "hyw_Armn": "Western Armenian",
    "wol_Latn": "Wolof",
    "yid_Hebr": "Yiddish",
    "say_Latn": "Zaar"
}
