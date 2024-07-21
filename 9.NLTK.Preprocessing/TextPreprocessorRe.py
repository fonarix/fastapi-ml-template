# RegEx

# https://docs.python.org/uk/3/library/re.html#text-munging

import logging
import re
import difflib

# Logger setup:
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('TppRe')

# Helpers:


def _CalculaterStringDifference(text_one, text_two):
    #difference = difflib.Differ()
    #text_one = "If rents are received later than five (5)"
    #text_two = "If rents are received later than eight (8)"

    n_text_one = text_one.replace(" ","\n")
    n_text_two = text_two.replace(" ","\n")

    diff = difflib.ndiff(n_text_one.splitlines(0), n_text_two.splitlines(0))

    one_lst = []
    two_lst = []

    for change in diff:
        if change[0] == "-":
            one_lst.append(change[2:])
        elif change[0] == "+":
            two_lst.append(change[2:])

    return one_lst, two_lst

###############################################################################
# Utils routines:

#
#Function to clean html tags from a sentence
def clean_html_tags(sentence):
    #logger.info("---------------------------------------------------")
    #logger.info("Removing Html")
    pattern = re.compile('<.*?>')
    cleaned_text = re.sub(pattern,' ',sentence)
    if sentence != cleaned_text:
        #logger.info("Removed Html:")
        #one_list, two_list = _CalculaterStringDifference(sentence, cleaned_text)
        #logger.info('Initial: %s', one_list)
        #logger.info('Updated: %s', two_list)
        pass
    return cleaned_text

# Remove Duplicate Spaces and Newline Characters Using the join() and split() Methods
def remove_whitespaces_duplicates(text):
    result_text = " ".join(text.split())
    return result_text


#Function to keep only words containing letters A-Z and a-z.
#this will remove all punctuations, special characters.
def remove_punct_leave_only_en_letters(sentence):
    cleaned_text  = re.sub('[^a-zA-Z]',' ',sentence)
    if sentence != cleaned_text:
        #logger.info("Removed punctuations:")
        #one_list, two_list = _CalculaterStringDifference(sentence, cleaned_text)
        #logger.info('Initial: %s', one_list)
        #logger.info('Updated: %s', two_list)
        pass
    return cleaned_text

# Directly remove punct
#def rem_pun_(sentence):
#    cleaned_text  = re.sub('[^a-zA-Z]',' ',sentence)
#    return (cleaned_text)

# Remove URL from sentences.
#def remove_urls(text, replacement_text="[URL REMOVED]"):
def remove_urls(text, replacement_text=""):
    # Define a regex pattern to match URLs
    #url_pattern = re.compile(r'https?://\S+|www\.\S+')
    #url_pattern = re.compile(r'^https?:\/\/.*[\r\n]*|www\.\S+')
    url_pattern = re.compile(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''')

    # Use the sub() method to replace URLs with the specified replacement text
    text_without_urls = url_pattern.sub(replacement_text, text)
    return text_without_urls

# Remove words like 'ddddddddd', 'funnnnnn', 'coolllllll' etc.
# Preserves words like 'goods', 'cool', 'best' etc. We will remove all such words which has three consecutive repeating characters.
def remove_extra(sen):
    cleaned_text  = re.sub("\s*\b(?=\w*(\w)\1{2,})\w*\b",' ',sen)
    return (cleaned_text)

#Convert all the words to lower case
#Source https://github.com/saugatapaul1010/Amazon-Fine-Food-Reviews-Analysis

def normalize_words_punctuation_and_lower_case(x):
    x = str(x).lower()
    x = x.replace(",000,000", " m").replace(",000", " k").replace("′", "'").replace("’", "'")\
                           .replace("won't", " will not").replace("cannot", " can not").replace("can't", " can not")\
                           .replace("n't", " not").replace("what's", " what is").replace("it's", " it is")\
                           .replace("'ve", " have").replace("'m", " am").replace("'re", " are")\
                           .replace("he's", " he is").replace("she's", " she is").replace("'s", " own")\
                           .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")\
                           .replace("€", " euro ").replace("'ll", " will").replace("how's"," how has").replace("y'all"," you all")\
                           .replace("o'clock"," of the clock").replace("ne'er"," never").replace("let's"," let us")\
                           .replace("finna"," fixing to").replace("gonna"," going to").replace("gimme"," give me").replace("gotta"," got to").replace("'d"," would")\
                           .replace("daresn't"," dare not").replace("dasn't"," dare not").replace("e'er"," ever").replace("everyone's"," everyone is")\
                           .replace("'cause'"," because")

    x = re.sub(r"([0-9]+)000000", r"\1m", x)
    x = re.sub(r"([0-9]+)000", r"\1k", x)
    return x

###############################################################################
#

'''

kReRemoveHtlmTags       = 're.RemoveHtlmTags'
kReRemoveUrls           = 're.RemoveUrls'
kReRemovePunct          = 're.RemovePunct'

def _GetPipelineDictRoutines():
    func_dict = {}
    func_dict[kReRemoveHtlmTags] = clean_html_tags
    func_dict[kReRemoveUrls] = remove_urls
    func_dict[kReRemovePunct] = remove_punct_leave_only_en_letters
    return func_dict


class _TextPreprocessorRe:
    def __init__(self):
        return

    def RunFunction(self, func_name, in_params):
        callbacks = _GetPipelineDictRoutines()

        func = callbacks.get(func_name)
        if func is not None:

            # functiojn overload for pd.Series
            if isinstance(in_params, pd.Series):
                result = in_params.apply(func)
                return result

            # else - apply raw data
            result = func(in_params)
            return result

        logger.error("Function ", func_name, " not found")
        return None


###############################################################################

# Demo and Debugging
if __name__ == '__main__':
    txtPreprocessr = TextPreprocessorRe()
    preprocessed = txtPreprocessr.RunFunction(kReRemoveHtlmTags, 'This is a demo testtext!<> Visit us at Website: https://www.geeksforgeeks.org/')
    logger.info(preprocessed)

'''


# Development and testing demo
if __name__ == '__main__':
    initial_text = '<a href="www.foo.com" class="bar">I Want This <b>text!</b></a>'

    result_no_urls = remove_urls(initial_text)
    print(result_no_urls)

    difference = difflib.Differ()
    #Calculates the difference
    #one_lst, two_lst = _CalculaterStringDifference(initial_text, result_no_urls)
    #print (one_lst)
    #print (two_lst)
    #print (''.join(diff))
    #print (diff)

    result = clean_html_tags(result_no_urls)
    print(result)

    result = remove_punct_leave_only_en_letters(result)
    print(result)

    result = remove_whitespaces_duplicates(result)
    print(result)


    #logger.info("Difference:")
    #logger.info(initial_text)
    #logger.info(result)
