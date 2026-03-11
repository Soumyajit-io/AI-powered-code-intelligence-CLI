from .language_based import lang, lang_based
from ...tools.scanner import scan_project
import os



def split_file(files):
    for file in files :
        _, ext = os.path.splitext(file)
        print(ext)
        print(file)
        if ext in lang:
            
            lang_based(file,ext)
        else:
            pass

if __name__=="__main__":
    split_file(scan_project(os.getcwd()))