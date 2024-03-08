# importing required modules
import textract
import glob
import os
from ebooklib import epub
from epub2txt import epub2txt

# file = 'texts/(Routledge Classics) Georg Simmel, David Frisby, Charles Lemert - The Philosophy of Money-Routledge (2011).epub'

# text = textract.process(file, extension="epub", input_encoding='utf8', output_encoding='utf8')
# print(text)

# res = epub2txt(file)
# print(res)

"""
Some books require textract, others work better with epub2txt, but I do not know exactly why
Fanged Noumena needed utf8 encoding when going from extraction to string, writing to txt file
capital v1, v2, v3 worked with epub2txt directly
"""

for file in glob.glob("texts/*"):

    path = file.split(".epub")[0]
    name = path.split("/")[-1]

    if os.path.isfile(f"data/{name}.txt"):
        pass

    else:
        print(file)

        text = epub2txt(file)
        # text = textract.process(file, extension="epub") # returns bytes string sometimes
        print(text)
        f = open(f"data/{name}.txt", "a")

        f.write(text)

        f.close()
