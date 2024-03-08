# importing required modules
from PyPDF2 import PdfReader
import glob
import os

for file in glob.glob("texts/*"):
    print(file)

    path = file.split(".pdf")[0]

    if os.path.exists(f"{path}.txt"):
        continue

    name = path.split("/")[-1]

    f = open(f"data/{name}.txt", "a")

    # creating a pdf reader object
    reader = PdfReader(file)

    # printing number of pages in pdf file
    print(len(reader.pages))

    # getting a specific page from the pdf file
    page = reader.pages[0]

    # extracting text from page
    text = page.extract_text()
    print(text)

    for i in range(len(reader.pages)):
        # getting a specific page from the pdf file
        page = reader.pages[i]

        # extracting text from page
        text = page.extract_text()

        f.write(text)

    f.close()

