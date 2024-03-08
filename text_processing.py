# importing required modules
from PyPDF2 import PdfReader

# creating a pdf reader object
reader = PdfReader('texts/TheDarkEnlightenment.pdf')

# printing number of pages in pdf file
print(len(reader.pages))

# getting a specific page from the pdf file
page = reader.pages[0]

# extracting text from page
text = page.extract_text()
print(text)

f = open("data/dataset.txt", "a")

for i in range(len(reader.pages)):
    # getting a specific page from the pdf file
    page = reader.pages[i]

    # extracting text from page
    text = page.extract_text()

    f.write(text)

f.close()


