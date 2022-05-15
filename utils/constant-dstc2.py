BERT_SEP=102

ALBERT_SEP=3

track_slots = [
    "area",
    "food",
    "pricerange"
]

ansvocab=[
    [
        "centre",
        "north",
        "west",
        "south",
        "east"
    ],
    [
        "afghan",
        "african",
        "afternoon tea",
        "asian oriental",
        "australasian",
        "australian",
        "austrian",
        "barbeque",
        "basque",
        "belgian",
        "bistro",
        "brazilian",
        "british",
        "canapes",
        "cantonese",
        "caribbean",
        "catalan",
        "chinese",
        "christmas",
        "corsica",
        "creative",
        "crossover",
        "cuban",
        "danish",
        "eastern european",
        "english",
        "eritrean",
        "european",
        "french",
        "fusion",
        "gastropub",
        "german",
        "greek",
        "halal",
        "hungarian",
        "indian",
        "indonesian",
        "international",
        "irish",
        "italian",
        "jamaican",
        "japanese",
        "korean",
        "kosher",
        "latin american",
        "lebanese",
        "light bites",
        "malaysian",
        "mediterranean",
        "mexican",
        "middle eastern",
        "modern american",
        "modern eclectic",
        "modern european",
        "modern global",
        "molecular gastronomy",
        "moroccan",
        "new zealand",
        "north african",
        "north american",
        "north indian",
        "northern european",
        "panasian",
        "persian",
        "polish",
        "polynesian",
        "portuguese",
        "romanian",
        "russian",
        "scandinavian",
        "scottish",
        "seafood",
        "singaporean",
        "south african",
        "south indian",
        "spanish",
        "sri lankan",
        "steakhouse",
        "swedish",
        "swiss",
        "thai",
        "the americas",
        "traditional",
        "turkish",
        "tuscan",
        "unusual",
        "vegetarian",
        "venetian",
        "vietnamese",
        "welsh",
        "world"

    ],
    [
        "cheap",
        "moderate",
        "expensive"
    ]
]

slot_map = {
    "area": 0,
    "food": 1,
    "pricerange": 2
}

n_slot = len(track_slots)

EXPERIMENT_DOMAINS = ["restaurant"]

domain2id = {d: i for i, d in enumerate(EXPERIMENT_DOMAINS)}

OP_SET = {
    '2': {'update': 0, 'carryover': 1},
    '3-1': {'update': 0, 'carryover': 1, 'dontcare': 2},
    '3-2': {'update': 0, 'carryover': 1, 'delete': 2},
    '4': {'delete': 0, 'update': 1, 'dontcare': 2, 'carryover': 3},
    '6': {'delete': 0, 'update': 1, 'dontcare': 2, 'carryover': 3, 'yes': 4, 'no': 5}
}

TURN_SPLIT = 8

TEST_TURN_SPLIT = 8