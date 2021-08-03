from find_lines import *
import os

directory = "documents"

for filename in os.listdir(directory):
    if filename.endswith("tif") or filename.endswith("TIF"):
        file = cv2.imread(f"documents/{filename}", cv2.IMREAD_GRAYSCALE)
        if file is not None:
            find_lines(file, filename)
    else:
        continue
