# Normalize class names across datasets
ACTION_CLASS_MAPPING = {
    'NTU': {
        "A009": "standing up",
        "A014": "wearing jacket",
        "A087": "put on bag",
        "A033": "look at wrist watch"
    },
    'PKU': {
        "4": "look at wrist watch",
        "34": "standing up",
        "48": "wearing jacket"
    },
    'Charades': {
        "c154": "standing up",
        "c148": "wearing jacket",
        "c023": "put on bag",
        "c092": "check surroundings"
    },
    'MA-52': {
        4 : 'standing up',
        8 : 'check surroundings',
        3 : 'turn body',
        6 : 'look at stop display'
    },
    'MMAct': {
        'looking_around': 'check surroundings',
        'checking_time': 'look at wrist watch',
        'standing_up' : 'standing up'
    }
}

DESIRED_CLASSES = [
    'standing up', 'wearing jacket', 'put on bag', 
    'check surroundings', 'look at wrist watch', 
    'turn body', 'look at stop display'
]