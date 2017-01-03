'''
Data load and pre-process
'''

def load_data(filename):
    contents =[]
    with open(filename, encoding="utf-8") as f:
        for line in f:
            title, content = line.strip().split(":")
            contents.append(content)
    return contents

