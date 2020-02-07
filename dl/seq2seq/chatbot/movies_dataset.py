import csv



def data_processing():
    stem = "../../data/cornell-movie-dialogs-corpus/"
    lines_path = stem + "movie_lines.txt"
    conversations_path = stem + "movie_conversations.txt"
    formatted_path = stem + "formatted_movie_lines.txt"

    qa_pairs = import_movie_data(lines_path, conversations_path)
    write_sentence_pairs_to_csv(formatted_path, qa_pairs) 


"""
    Example data in movie_lines.txt
    L1045 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ They do not!
    [LineId]      [CharId]   [MovieId]  [CharName]     [Dialog]
"""
def import_movie_data(lines_path, conversations_path):
    LINE_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
    CONVERSATION_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]
    
    lines = load_lines(lines_path, LINE_FIELDS)
    conversations = load_conversations(conversations_path, lines, CONVERSATION_FIELDS)
    qa_pairs = extract_sentence_pairs(conversations)
    return qa_pairs

    
"""
    Internal representation of lines is a dictionary i.e. object. 
    Example entry: 
    {
        'lineID': 'L1045', 
        'characterID': 'u0', 
        'movieID': 'm0', 
        'character': 'BIANCA', 
        'text': 'They do not!\n'
    }
"""
def load_lines(file_path, fields):
    lines = {}
    with open(file_path, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            lineObj = {}
            for i, field in enumerate(fields):
                lineObj[field] = values[i]
            lines[lineObj['lineID']] = lineObj
    return lines


"""
    Internal representation of conversations is a list of dictionaries.
    Example element:
    {
        'character1ID': 'u0', 
        'character2ID': 'u2', 
        'movieID': 'm0', 
        'utteranceIDs': "['L194', 'L195', 'L196', 'L197']\n", 
        'lines': [
            {'lineID': 'L194', 'characterID': 'u0', 'movieID': 'm0', 'character': 'BIANCA', 
                'text': 'Can we make this quick?  Roxanne Korrine and Andrew Barrett are having an incredibly horrendous public break- up on the quad.  Again.\n'}, 
            {'lineID': 'L195', 'characterID': 'u2', 'movieID': 'm0', 'character': 'CAMERON', 
                'text': "Well, I thought we'd start with pronunciation, if that's okay with you.\n"}, 
            {'lineID': 'L196', 'characterID': 'u0', 'movieID': 'm0', 'character': 'BIANCA', 
                'text': 'Not the hacking and gagging and spitting part.  Please.\n'}, 
            {'lineID': 'L197', 'characterID': 'u2', 'movieID': 'm0', 'character': 'CAMERON', 
                'text': "Okay... then how 'bout we try out some French cuisine.  Saturday?  Night?\n"}
            ]
    }
"""
def load_conversations(file_path, lines, fields):
    conversations = []
    with open(file_path, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            conv_obj = {}
            for i, field in enumerate(fields):
                conv_obj[field] = values[i]
            # Convert string to list (convObj["utteranceIDs"] == "['L598485', 'L598486', ...]")
            lineIds = eval(conv_obj["utteranceIDs"])
            # Reassemble lines
            conv_obj["lines"] = []
            for lineId in lineIds:
                conv_obj["lines"].append(lines[lineId])
            conversations.append(conv_obj)
    return conversations


"""
    Internal representation of question answer pairs is a list.
    The q_a is a list of 2 elements: [Question string, Answer string]
    Example entry: 
    ['Can we make this quick?  Roxanne Korrine and Andrew Barrett are having an incredibly horrendous public break- up on the quad.  Again.', 
    "Well, I thought we'd start with pronunciation, if that's okay with you."]

"""
def extract_sentence_pairs(conversations):
    qa_pairs = []
    for conversation in conversations:
        # Iterate over all the lines of the conversation
        for i in range(len(conversation["lines"]) - 1):  # We ignore the last line (no answer for it)
            input_line = conversation["lines"][i]["text"].strip()
            target_line = conversation["lines"][i+1]["text"].strip()
            # Filter wrong samples (if one of the lists is empty)
            if input_line and target_line:
                qa_pairs.append([input_line, target_line])
    return qa_pairs

def write_sentence_pairs_to_csv(datafile, sentence_pairs):
    print("\nWriting newly formatted file...")
    delimiter = '\t'
    with open(datafile, 'w', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter=delimiter)
        for pair in sentence_pairs:
            writer.writerow(pair)


# testing helper functions
if __name__ == '__main__':
    data_processing()