import os.path
raw_documents = []
snippets = []
with open( os.path.join("dataset", "train.csv") ,"r") as fin:
    for line in fin.readlines():
        text = line.strip()
        raw_documents.append( text )
        # keep a short snippet of up to 100 characters as a title for each article
        snippets.append( text[0:min(len(text),100)] )
print raw_documents
