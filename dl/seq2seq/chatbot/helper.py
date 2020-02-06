"""
    Example data in movie_lines.txt
    L1045 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ They do not!
    [LineId]      [CharId]   [MovieId]  [CharName]     [Dialog]
"""


def import_movie_data():
    lines_path = "../../data/cornell-movie-dialogs-corpus/movie_lines.txt"
    conversations_path = "../../data/cornell-movie-dialogs-corpus/movie_conversations.txt"

    LINE_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
    CONVERSATION_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]
    
    lines = loadLines(lines_path, LINE_FIELDS)
    conversations = loadConversations(conversations_path, lines, CONVERSATION_FIELDS)
    
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
def loadLines(file_path, fields):
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
def loadConversations(file_path, lines, fields):
    conversations = []
    with open(file_path, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            convObj = {}
            for i, field in enumerate(fields):
                convObj[field] = values[i]
            # Convert string to list (convObj["utteranceIDs"] == "['L598485', 'L598486', ...]")
            lineIds = eval(convObj["utteranceIDs"])
            # Reassemble lines
            convObj["lines"] = []
            for lineId in lineIds:
                convObj["lines"].append(lines[lineId])
            conversations.append(convObj)
    return conversations


# testing helper functions
if __name__ == '__main__':
    import_movie_data()