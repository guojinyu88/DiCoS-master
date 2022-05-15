BERT_SEP=102

ALBERT_SEP=3

track_slots = [
    "category",
    "date",
    "location",
    "meal",
    "num_people",
    "price_range",
    "rating",
    "restaurant_name",
    "time"
]

ansvocab=[
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    []
]

slot_map = {
    "category": 0,
    "date": 1,
    "location": 2,
    "meal": 3,
    "num_people": 4,
    "price_range": 5,
    "rating": 6,
    "restaurant_name": 7,
    "time": 8
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