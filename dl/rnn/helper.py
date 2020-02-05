import unidecode
import string
import random
import re

def import_and_sanitize(filepath):
    all_characters = string.printable
    n_characters = len(all_characters)
    current_file = unidecode.unidecode(open(filepath).read())
    file_len = len(current_file)
    print("successfully imported file {} with length = {}", filepath, file_len)

