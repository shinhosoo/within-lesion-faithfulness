import os


# PROJECT_ROOT = the directory that contains this package (SkinCancer/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# defaults are relative to the project root; override with environment variables
DATA_ROOT = os.environ.get(
    "HAM_DATA_ROOT", os.path.join(PROJECT_ROOT, "data", "HAM10000_extracted")
)
METADATA_CSV = os.path.join(DATA_ROOT, "HAM10000_metadata.csv")


def _resolve_image_dirs(root):
    """Find the part_1 / part_2 image folders case-insensitively."""
    found = []
    if os.path.isdir(root):
        entries = {name.lower(): name for name in os.listdir(root)}
        for part in ("ham10000_images_part_1", "ham10000_images_part_2"):
            if part in entries:
                found.append(os.path.join(root, entries[part]))
    if not found:  # fall back to the original hard-coded names when neither folder is found
        found = [
            os.path.join(root, "HAM10000_images_part_1"),
            os.path.join(root, "ham10000_images_part_2"),
        ]
    return found


IMAGE_DIRS = _resolve_image_dirs(DATA_ROOT)

SPLIT_ROOT = os.environ.get("HAM_SPLIT_ROOT", os.path.join(PROJECT_ROOT, "splits"))

# test split for the seven-class evaluation
FINAL7_TEST_CSV = os.path.join(SPLIT_ROOT, "global_test.csv")

STAGE2_FINAL_MAPPING = {
    "nv": 0,
    "mel": 1,
    "bkl": 2,
    "df": 3,
    "vasc": 4,
    "bcc": 5,
    "akiec": 6,
}

FINAL_CLASS_NAMES = ["nv", "mel", "bkl", "df", "vasc", "bcc", "akiec"]
MEL_CLASS_NAMES = ["nv", "mel"]
NM_CLASS_NAMES = ["bkl", "df", "vasc", "bcc", "akiec"]
