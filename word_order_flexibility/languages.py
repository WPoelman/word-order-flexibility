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
    "Armenian": [
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
    "Chinese": [
        "UD_Chinese-CFL",
        "UD_Chinese-HK",
        "UD_Chinese-GSD",
        "UD_Chinese-PatentChar",
        "UD_Chinese-Beginner",
        "UD_Chinese-PUD",
        "UD_Chinese-GSDSimp",
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
        "UD_Czech-Poetry",
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
        "UD_English-ESLSpok",
        "UD_English-GENTLE",
        # "UD_English-GUMReddit",  # redacted (all forms are underscores)
        "UD_English-ParTUT",
        "UD_English-CTeTex",
        "UD_English-PUD",
        "UD_English-CHILDES",
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
        "UD_Faroese-FarPaHC",
        "UD_Faroese-OFT",
    ],
    "Finnish": [
        "UD_Finnish-FTB",
        "UD_Finnish-TDT",
        "UD_Finnish-PUD",
        "UD_Finnish-OOD",
    ],
    "French": [
        "UD_French-ParisStories",
        "UD_French-Rhapsodie",
        "UD_French-ParTUT",
        "UD_French-PUD",
        "UD_French-PoitevinDIVITAL",
        "UD_French-FQB",
        "UD_French-Sequoia",
        "UD_French-GSD",
        "UD_French-ALTS",
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
        "UD_German-LIT",
        "UD_German-PUD",
        "UD_German-GSD",
    ],
    "Gothic": [
        "UD_Gothic-PROIEL",
    ],
    "Modern Greek": [
        "UD_Greek-Lesbian",
        "UD_Greek-GUD",
        "UD_Greek-GDT",
        "UD_Greek-Messinian",
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
        "UD_Icelandic-IcePaHC",
    ],
    "Indonesian": [
        "UD_Indonesian-CSUI",
        "UD_Indonesian-PUD",
        "UD_Indonesian-GSD",
    ],
    "Irish": [
        "UD_Irish-IDT",
        "UD_Irish-Cadhan",
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
        "UD_Italian-KIParlaForest",
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
    "Norwegian": [
        "UD_Norwegian-Nynorsk",
        "UD_Norwegian-Bokmaal",
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
        "UD_Polish-MPDT",
    ],
    "Pomak": [
        "UD_Pomak-Philotis",
    ],
    "Portuguese": [
        "UD_Portuguese-DANTEStocks",
        "UD_Portuguese-PetroGold",
        "UD_Portuguese-Porttinari",
        "UD_Portuguese-Bosque",
        "UD_Portuguese-PUD",
        "UD_Portuguese-CINTIL",
        "UD_Portuguese-GSD",
    ],
    "Romanian": [
        "UD_Romanian-Nonstandard",
        "UD_Romanian-ArT",
        "UD_Romanian-TueCL",
        "UD_Romanian-RRT",
        "UD_Romanian-SiMoNERo",
        "UD_Romanian-MolDoRo",
    ],
    "Russian": [
        "UD_Russian-PUD",
        "UD_Russian-Taiga",
        "UD_Russian-SynTagRus",
        "UD_Russian-Poetry",
        "UD_Russian-GSD",
    ],
    "Sanskrit": [
        "UD_Sanskrit-Vedic",
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
        "UD_Slovenian-SST",
        "UD_Slovenian-SSJ",
    ],
    "Spanish": [
        "UD_Spanish-GSD",
        "UD_Spanish-AnCora",
        "UD_Spanish-COSER",
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
