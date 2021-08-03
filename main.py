from find_lines import *
import os
import xml.etree.ElementTree as et


def remove_words(words, image):
    for word in words:
        start_point = (int(word.attrib["left"]), int(word.attrib["top"]))
        end_point = tuple(map(lambda x, y: x + y, start_point,
                              (int(word.attrib["width"]), int(word.attrib["height"]))))
        image = cv2.rectangle(image, start_point, end_point, (255, 255, 255), -1)
    return image


directory = "preprocessed_documents"
og_directory = "documents"

for filename in os.listdir(og_directory):
    if filename.casefold().endswith("tif"):
        xdocfilename = filename.casefold().replace(".tif", ".xdc")
        with gzip.open(f"{og_directory}/{xdocfilename}", mode='r') as f:
            xml_text = f.read()
        xml = et.fromstring(xml_text)
        words = xml.findall(".//word")
        boxes = xml.findall(".//box")
        image = cv2.imread(f"documents/{filename}", cv2.IMREAD_COLOR)
        image = remove_words(boxes, remove_words(words, image))
        cv2.imwrite(f"preprocessed_documents/{filename}", image)

for filename in os.listdir(directory):
    if filename.casefold().endswith("tif"):
        file = cv2.imread(f"{directory}/{filename}", cv2.IMREAD_GRAYSCALE)
        og_file = cv2.imread(f"{og_directory}/{filename}", cv2.IMREAD_COLOR)
        if file is not None:
            find_lines(file, filename, og_file)



