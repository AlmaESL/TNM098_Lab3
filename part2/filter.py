from preprocessing import load_and_process

keywords = ['attack', 'terror', 'bomb', 'terrorism', 'dead', 'injured', 'threat', 'violence', 'explosion', 'shooting', 'casualties', 
            'assault', 'hostage', 'crisis', 'emergency', 'safety', 'security']

def is_relevant(content): 
    # return any(word in content for word in keywords)
    # filter on keywords strictly
    return any(word in content for word in keywords)
    

def get_relevant(content): 

    content['relevant'] = content['processed_content'].apply(is_relevant)
    content = content[content['relevant'] == True]
    return content
    