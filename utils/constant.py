BERT_SEP=102

ALBERT_SEP=3

track_slots = [
    "attraction-area",
    "attraction-name",
    "attraction-type",
    "hotel-area",
    "hotel-bookday",
    "hotel-bookpeople",
    "hotel-bookstay",
    "hotel-internet",
    "hotel-name",
    "hotel-parking",
    "hotel-pricerange",
    "hotel-stars",
    "hotel-type",
    "restaurant-area",
    "restaurant-bookday",
    "restaurant-bookpeople",
    "restaurant-booktime",
    "restaurant-food",
    "restaurant-name",
    "restaurant-pricerange",
    "taxi-arriveby",
    "taxi-departure",
    "taxi-destination",
    "taxi-leaveat",
    "train-arriveby",
    "train-bookpeople",
    "train-day",
    "train-departure",
    "train-destination",
    "train-leaveat"
]

ansvocab = [
    [
        "cheap",
        "expensive",
        "moderate"
    ],
    [
        "centre",
        "east",
        "north",
        "south",
        "west"
    ],
    [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "15"
    ],
    [
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    ],
    [
        "yes",
        "no"
    ],
    [
        "birmingham new street",
        "bishops stortford",
        "broxbourne",
        "cambridge",
        "ely",
        "kings lynn",
        "leicester",
        "london kings cross",
        "london liverpool street",
        "norwich",
        "peterborough",
        "stansted airport",
        "stevenage"
    ],
    [
        "guesthouse",
        "hotel"
    ],
    [
        "architecture",
        "boat",
        "cinema",
        "college",
        "concerthall",
        "entertainment",
        "museum",
        "multiple sports",
        "nightclub",
        "park",
        "swimmingpool",
        "theatre"
    ],
    []
]

slot_map = {
    "attraction-area": 1,
    "attraction-name": 8,
    "attraction-type": 7,
    "hotel-area": 1,
    "hotel-bookday": 3,
    "hotel-bookpeople": 2,
    "hotel-bookstay": 2,
    "hotel-internet": 4,
    "hotel-name": 8,
    "hotel-parking": 4,
    "hotel-pricerange": 0,
    "hotel-stars": 2,
    "hotel-type": 6,
    "restaurant-area": 1,
    "restaurant-bookday": 3,
    "restaurant-bookpeople": 2,
    "restaurant-booktime": 8,
    "restaurant-food": 8,
    "restaurant-name": 8,
    "restaurant-pricerange": 0,
    "taxi-arriveby": 8,
    "taxi-departure": 8,
    "taxi-destination": 8,
    "taxi-leaveat": 8,
    "train-arriveby": 8,
    "train-bookpeople": 2,
    "train-day": 2,
    "train-departure": 5,
    "train-destination": 5,
    "train-leaveat": 8
}

n_slot = len(track_slots)

EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi", "bus", "hospital", "police"]

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