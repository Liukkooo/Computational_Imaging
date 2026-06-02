from pathlib import Path

BASE_DIR = Path(__file__).parent
OUTPUT_DIR_BLUR = BASE_DIR / "dataset" / "blur"
OUTPUT_DIR_BLUR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_BLUR = str(OUTPUT_DIR_BLUR)

OUTPUT_DIR_NOISE = BASE_DIR / "dataset" / "noise"
OUTPUT_DIR_NOISE.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_NOISE = str(OUTPUT_DIR_NOISE)