"""
This code is from https://github.com/jasonwu0731/trade-dst
"""


def fix_general_label_error(labels, type, slots):
    delslot=[]
    for sl in labels:
        name=sl['slots'][0][0]
        if name not in slots.keys():
            delslot.append(sl)
            continue
        if not isinstance(sl['slots'][0][1],list):
            sl['slots'][0][1]=[sl['slots'][0][1]]
        if " " in name:
            newname=name.replace(" ","")
            sl['slots'][0]=[newname,sl['slots'][0][1]]
    for s in delslot:
        labels.remove(s)
    label_dict = dict([ (l[0], l[1]) for l in labels]) if type else dict([ (l["slots"][0][0], l["slots"][0][1]) for l in labels])

    GENERAL_TYPO = {
        # type
        "guesthouse":"guest house", "guesthouses":"guest house", "guest":"guest house", "mutiple sports":"multiple sports", 
        "sports":"multiple sports", "mutliple sports":"multiple sports","swimmingpool":"swimming pool", "concerthall":"concert hall", 
        "concert":"concert hall", "pool":"swimming pool", "night club":"nightclub", "mus":"museum", "ol":"architecture", 
        "colleges":"college", "coll":"college", "architectural":"architecture", "musuem":"museum", "churches":"church",
        # area
        "center":"centre", "center of town":"centre", "near city center":"centre", "in the north":"north", "cen":"centre", "east side":"east", 
        "east area":"east", "west part of town":"west", "ce":"centre",  "town center":"centre", "centre of cambridge":"centre", 
        "city center":"centre", "the south":"south", "scentre":"centre", "town centre":"centre", "in town":"centre", "north part of town":"north", 
        "centre of town":"centre", "cb30aq": "none",
        # price
        "mode":"moderate", "moderate -ly": "moderate", "mo":"moderate", 
        # day
        "next friday":"friday", "monda": "monday", 
        # parking
        "free parking":"free",
        # internet
        "free internet":"yes",
        # star
        "4 star":"4", "4 stars":"4", "0 star rarting":"none",
        # others 
        "y":"yes", "any":"dontcare", "n":"no", "does not care":"dontcare", "not men":"none", "not":"none", "not mentioned":"none",
        '':"none", "not mendtioned":"none", "3 .":"3", "does not":"no", "fun":"none", "art":"none",  
        }

    for slot in slots:
        if slot in label_dict.keys():
            # general typos
            for a in label_dict[slot]:
                if a in GENERAL_TYPO.keys():
                    label_dict[slot].remove(a)
                    label_dict[slot].append(a.replace(a, GENERAL_TYPO[a]))

            # miss match slot and value 
                if  slot == "hotel-type" and a in ["nigh", "moderate -ly priced", "bed and breakfast", "centre", "venetian", "intern", "a cheap -er hotel"] or \
                    slot == "hotel-internet" and a == "4" or \
                    slot == "hotel-pricerange" and a == "2" or \
                    slot == "attraction-type" and a in ["gastropub", "la raza", "galleria", "gallery", "science", "m"] or \
                    "area" in slot and a in ["moderate"] or \
                    "day" in slot and a == "t":
                    label_dict[slot].remove(a)
                    # label_dict[slot].append('none')
                elif slot == "hotel-type" and a in ["hotel with free parking and free wifi", "4", "3 star hotel"]:
                    label_dict[slot].remove(a)
                    label_dict[slot].append(["hotel"])
                elif slot == "hotel-star" and a == "3 star hotel":
                    label_dict[slot].remove(a)
                    label_dict[slot].append('3')
                elif "area" in slot:
                    if a == "no":
                        label_dict[slot].remove(a)
                        label_dict[slot].append("north")
                    elif a == "we":
                        label_dict[slot].remove(a)
                        label_dict[slot].append("west")
                    elif a == "cent":
                        label_dict[slot].remove(a)
                        label_dict[slot].append("centre")
                    elif "day" in slot:
                        if a == "we":
                            label_dict[slot].remove(a)
                            label_dict[slot].append("wednesday")
                        elif a == "no":
                            label_dict[slot].remove(a)
                            # label_dict[slot].append("none")
                    elif "price" in slot and a == "ch":
                        label_dict[slot].remove(a)
                        label_dict[slot].append("cheap")
                    elif "internet" in slot and a == "free":
                        label_dict[slot].remove(a)
                        label_dict[slot].append("yes")

                # some out-of-define classification slot values
                    if  slot == "restaurant-area" and a in ["stansted airport", "cambridge", "silver street"] or \
                        slot == "attraction-area" and a in ["norwich", "ely", "museum", "same area as hotel"]:
                        label_dict[slot].remove(a)
                        # label_dict[slot].append("none")
        else:
            label_dict[slot]=[]
    return label_dict

#2.2
# def fix_general_label_error(labels, type, slots):
#     label_dict = dict([(l[0], l[1]) for l in labels]) if type else dict(
#         [(l["slots"][0][0], l["slots"][0][1]) for l in labels])
#
#     GENERAL_TYPO = {
#         # type
#         "guesthouse": "guest house", "guesthouses": "guest house", "guest": "guest house",
#         "mutiple sports": "multiple sports",
#         "sports": "multiple sports", "mutliple sports": "multiple sports", "swimmingpool": "swimming pool",
#         "concerthall": "concert hall",
#         "concert": "concert hall", "pool": "swimming pool", "night club": "nightclub", "mus": "museum",
#         "ol": "architecture",
#         "colleges": "college", "coll": "college", "architectural": "architecture", "musuem": "museum",
#         "churches": "church",
#         # area
#         "center": "centre", "center of town": "centre", "near city center": "centre", "in the north": "north",
#         "cen": "centre", "east side": "east",
#         "east area": "east", "west part of town": "west", "ce": "centre", "town center": "centre",
#         "centre of cambridge": "centre",
#         "city center": "centre", "the south": "south", "scentre": "centre", "town centre": "centre",
#         "in town": "centre", "north part of town": "north",
#         "centre of town": "centre", "cb30aq": "none",
#         # price
#         "mode": "moderate", "moderate -ly": "moderate", "mo": "moderate",
#         # day
#         "next friday": "friday", "monda": "monday",
#         # parking
#         "free parking": "free",
#         # internet
#         "free internet": "yes",
#         # star
#         "4 star": "4", "4 stars": "4", "0 star rarting": "none",
#         # others
#         "y": "yes", "any": "dontcare", "n": "no", "does not care": "dontcare", "not men": "none", "not": "none",
#         "not mentioned": "none",
#         '': "none", "not mendtioned": "none", "3 .": "3", "does not": "no", "fun": "none", "art": "none",
#     }
#
#     for slot in slots:
#         if slot in label_dict.keys():
#             # general typos
#             for a in label_dict[slot]:
#                 if a in GENERAL_TYPO.keys():
#                     label_dict[slot].remove(a)
#                     label_dict[slot].append(a.replace(a, GENERAL_TYPO[a]))
#
#                 # miss match slot and value
#                 if slot == "hotel-type" and a in ["nigh", "moderate -ly priced", "bed and breakfast", "centre",
#                                                   "venetian", "intern", "a cheap -er hotel"] or \
#                         slot == "hotel-internet" and a == "4" or \
#                         slot == "hotel-pricerange" and a == "2" or \
#                         slot == "attraction-type" and a in ["gastropub", "la raza", "galleria", "gallery", "science",
#                                                             "m"] or \
#                         "area" in slot and a in ["moderate"] or \
#                         "day" in slot and a == "t":
#                     label_dict[slot] = ["none"]
#             elif slot == "hotel-type" and label_dict[slot] in ["hotel with free parking and free wifi", "4",
#                                                                "3 star hotel"]:
#             label_dict[slot] = "hotel"
#         elif slot == "hotel-star" and label_dict[slot] == "3 star hotel":
#             label_dict[slot] = "3"
#         elif "area" in slot:
#             if label_dict[slot] == "no":
#                 label_dict[slot] = "north"
#             elif label_dict[slot] == "we":
#                 label_dict[slot] = "west"
#             elif label_dict[slot] == "cent":
#                 label_dict[slot] = "centre"
#         elif "day" in slot:
#             if label_dict[slot] == "we":
#                 label_dict[slot] = "wednesday"
#             elif label_dict[slot] == "no":
#                 label_dict[slot] = "none"
#         elif "price" in slot and label_dict[slot] == "ch":
#             label_dict[slot] = "cheap"
#         elif "internet" in slot and label_dict[slot] == "free":
#             label_dict[slot] = "yes"
#
#         # some out-of-define classification slot values
#         if slot == "restaurant-area" and label_dict[slot] in ["stansted airport", "cambridge", "silver street"] or \
#                 slot == "attraction-area" and label_dict[slot] in ["norwich", "ely", "museum", "same area as hotel"]:
#             label_dict[slot] = "none"
#
#
# return label_dict
#
# def fix_general_label_error_ontology(label_dict,slot_meta):
#     GENERAL_TYPO = {
#         # type
#         "guesthouse": "guest house", "guesthouses": "guest house", "guest": "guest house",
#         "mutiple sports": "multiple sports",
#         "sports": "multiple sports", "mutliple sports": "multiple sports", "swimmingpool": "swimming pool",
#         "concerthall": "concert hall",
#         "concert": "concert hall", "pool": "swimming pool", "night club": "nightclub", "mus": "museum",
#         "ol": "architecture",
#         "colleges": "college", "coll": "college", "architectural": "architecture", "musuem": "museum",
#         "churches": "church",
#         # area
#         "center": "centre", "center of town": "centre", "near city center": "centre", "in the north": "north",
#         "cen": "centre", "east side": "east",
#         "east area": "east", "west part of town": "west", "ce": "centre", "town center": "centre",
#         "centre of cambridge": "centre",
#         "city center": "centre", "the south": "south", "scentre": "centre", "town centre": "centre",
#         "in town": "centre", "north part of town": "north",
#         "centre of town": "centre", "cb30aq": "none",
#         # price
#         "mode": "moderate", "moderate -ly": "moderate", "mo": "moderate",
#         # day
#         "next friday": "friday", "monda": "monday",
#         # parking
#         "free parking": "free",
#         # internet
#         "free internet": "yes",
#         # star
#         "4 star": "4", "4 stars": "4", "0 star rarting": "none",
#         # others
#         "y": "yes", "any": "dontcare", "n": "no", "does not care": "dontcare", "not men": "none", "not": "none",
#         "not mentioned": "none",
#         '': "none", "not mendtioned": "none", "3 .": "3", "does not": "no", "fun": "none", "art": "none",
#     }
#
#     for slot in slot_meta:
#         # general typos
#         for v in slot:
#             if v in GENERAL_TYPO.keys():
#
#         if label_dict[slot] in GENERAL_TYPO.keys():
#             label_dict[slot] = label_dict[slot].replace(label_dict[slot], GENERAL_TYPO[label_dict[slot]])
#
#         # miss match slot and value
#         if slot == "hotel-type" and label_dict[slot] in ["nigh", "moderate -ly priced", "bed and breakfast",
#                                                          "centre", "venetian", "intern", "a cheap -er hotel"] or \
#                 slot == "hotel-internet" and label_dict[slot] == "4" or \
#                 slot == "hotel-pricerange" and label_dict[slot] == "2" or \
#                 slot == "attraction-type" and label_dict[slot] in ["gastropub", "la raza", "galleria", "gallery",
#                                                                    "science", "m"] or \
#                 "area" in slot and label_dict[slot] in ["moderate"] or \
#                 "day" in slot and label_dict[slot] == "t":
#             label_dict[slot] = "none"
#         elif slot == "hotel-type" and label_dict[slot] in ["hotel with free parking and free wifi", "4",
#                                                            "3 star hotel"]:
#             label_dict[slot] = "hotel"
#         elif slot == "hotel-star" and label_dict[slot] == "3 star hotel":
#             label_dict[slot] = "3"
#         elif "area" in slot:
#             if label_dict[slot] == "no":
#                 label_dict[slot] = "north"
#             elif label_dict[slot] == "we":
#                 label_dict[slot] = "west"
#             elif label_dict[slot] == "cent":
#                 label_dict[slot] = "centre"
#         elif "day" in slot:
#             if label_dict[slot] == "we":
#                 label_dict[slot] = "wednesday"
#             elif label_dict[slot] == "no":
#                 label_dict[slot] = "none"
#         elif "price" in slot and label_dict[slot] == "ch":
#             label_dict[slot] = "cheap"
#         elif "internet" in slot and label_dict[slot] == "free":
#             label_dict[slot] = "yes"
#
#         # some out-of-define classification slot values
#         if slot == "restaurant-area" and label_dict[slot] in ["stansted airport", "cambridge", "silver street"] or \
#                 slot == "attraction-area" and label_dict[slot] in ["norwich", "ely", "museum",
#                                                                    "same area as hotel"]:
#             label_dict[slot] = "none"
#
#     return label_dict
