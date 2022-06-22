def write(fpath, text):
    with open(fpath, "a+") as f:
        f.write(text + '\n')