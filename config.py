RESOURCES_FOLDER = "resources"
GITHUB_RELEASE_URL = "https://github.com/fschmid56/PretrainedSED/releases/download/v0.0.1/"

# checkpoints
CHECKPOINT_URLS = {}

# strong
CHECKPOINT_URLS['BEATs_strong_1'] = GITHUB_RELEASE_URL + "BEATs_strong_1.pt"
CHECKPOINT_URLS['ATST-F_strong_1'] = GITHUB_RELEASE_URL + "ATST-F_strong_1.pt"
CHECKPOINT_URLS['ASIT_strong_1'] = GITHUB_RELEASE_URL + "ASIT_strong_1.pt"
CHECKPOINT_URLS['fpasst_strong_1'] = GITHUB_RELEASE_URL + "fpasst_strong_1.pt"
CHECKPOINT_URLS['M2D_strong_1'] = GITHUB_RELEASE_URL + "M2D_strong_1.pt"
for width in ['06', '10']:
    CHECKPOINT_URLS[f'frame_mn{width}_strong_1'] = GITHUB_RELEASE_URL + f'frame_mn{width}_strong_1.pt'

# weak
CHECKPOINT_URLS['BEATs_weak'] = GITHUB_RELEASE_URL + "BEATs_weak.pt"
CHECKPOINT_URLS['ATST-F_weak'] = GITHUB_RELEASE_URL + "ATST-F_weak.pt"
CHECKPOINT_URLS['ASIT_weak'] = GITHUB_RELEASE_URL + "ASIT_weak.pt"
CHECKPOINT_URLS['fpasst_weak'] = GITHUB_RELEASE_URL + "fpasst_weak.pt"
CHECKPOINT_URLS['M2D_weak'] = GITHUB_RELEASE_URL + "M2D_weak.pt"

# ssl
CHECKPOINT_URLS['BEATs_ssl'] = GITHUB_RELEASE_URL + "BEATs_ssl.pt"
CHECKPOINT_URLS['ATST-F_ssl'] = GITHUB_RELEASE_URL + "ATST-F_ssl.pt"
CHECKPOINT_URLS['ASIT_ssl'] = GITHUB_RELEASE_URL + "ASIT_ssl.pt"
CHECKPOINT_URLS['fpasst_ssl'] = GITHUB_RELEASE_URL + "fpasst_ssl.pt"
CHECKPOINT_URLS['M2D_ssl'] = GITHUB_RELEASE_URL + "M2D_ssl.pt"


# -----------------------------------------------------------------------------------------------

SED_TAGS = {
    "smeh": [
        "Belly laugh",
        "Chuckle, chortle",
        "Giggle",
        "Laughter",
        "Snicker",
    ],
    "aplavz": [
        "Applause",
        "Clapping",
        "Cheering",
    ],
    "dihanje": [
        "Breathing",
        "Respiratory sounds",
        "Pant",
        "Gasp",
        "Sigh",
        "Wheeze",
        "Sniff",
    ],
    "medmet": [
        "medmet",
    ],
}

SED_REVERSE_TAGS = {
    value: key
    for key, values in SED_TAGS.items()
    for value in values
}


SED_MODEL_NAME = "ATST" # ASTS, BEATS

SED_THRESHOLD = 0.4 # 0.05, 0.1, 0.2, 0.3, 0.5

SED_MEDIAN_FILTER = 3 # 3, 6, 9, 12, 15
