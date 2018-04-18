import os

def log(file, data):
    if os.path.exists(file):
        append_write = 'a'
    else:
        append_write = 'w'

    f = open(file, append_write)
    f.write(data + '\n')
    f.close()