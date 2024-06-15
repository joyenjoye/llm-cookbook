import difflib

def find_common_substring_indices(text1, text2):
    # Using difflib to find the longest common substring
    matcher = difflib.SequenceMatcher(None, text1, text2)
    match = matcher.find_longest_match(0, len(text1), 0, len(text2))

    if match:
        start_index = match.a
        end_index = match.a + match.size
        return start_index, end_index
    else:
        return None, None

def highlight_doc(doc, splited_node1, splited_node2):
    start_idx, end_idx = find_common_substring_indices(splited_node1, splited_node2)
    
    if start_idx is not None and end_idx is not None:
        # ANSI color codes for highlighting
        PURPLE = '\033[95m'
        GREEN = '\033[92m'
        BLUE = '\033[94m'
        RESET = '\033[0m'
        
        # Constructing the highlighted document
        highlighted_doc = (PURPLE + doc[:start_idx] + RESET + 
                           GREEN + doc[start_idx:end_idx] + RESET + 
                           BLUE + doc[end_idx:] + RESET)
        
        print(highlighted_doc)
    else:
        print("No common substring found.")