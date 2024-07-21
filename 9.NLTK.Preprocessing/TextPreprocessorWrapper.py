# https://lms.ithillel.ua/groups/65ccf134073f165247467f0c/lessons/65ccf134073f165247467f22

import logging
# Logger setup:
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('TextPreprocessorWrapper')

import pandas as pd
import string
import TextPreprocessorNltk as tpnltk
import TextPreprocessorRe as tpregex

#def _RemovePunctuation(text):
#    punctuaton_free = "".join([item for item in text if item not in string.punctuation])
#    return punctuaton_free

# lowecase
def LowerCaseTokens(tokens):
    lowercased_tokens = [token.lower() for token in tokens]
    return lowercased_tokens


def RemovePunktuationTokens(tokens):
    punctuated_text = [token for token in tokens if token not in string.punctuation]
    return punctuated_text


###############################################################################

# nltk:
kNltkSentencePunktTokenize      = 'nltk.SentencePunktTokenize'
kNltkSentenceTokenize           = 'nltk.SentenceTokenize'
kNltkWordTokenize               = 'nltk.WordTokenize'
kNltkLowerCaseTokens            = 'nltk.LowerCaseTokens'
kNltkPosTagging                 = 'nltk.PosTagging'
kNltkRemoveRareWords            = 'nltk.RemoveRareWords'

kNltkRemoveStopWords            = 'nltk.RemoveStopWords'

kNltkRemovePunktuationTokens    = 'nltk.RemovePunktuationTokens'
#kNltkRemovePunktuations         = 'nltk.RemovePunktuations'

# RegEx routines:
kReRemoveHtlmTags               = 're.RemoveHtlmTags'
kReRemoveUrls                   = 're.RemoveUrls'
kReNormalizePunctuationAndLower = 're.NormalizePunctuationAndLower'
kReRemoveTextPunctuation        = 're.RemoveTextPunctuation'
kReRemoveWhiteSpacesDuplicates  = 're.RemoveWhiteSpacesDuplicates'



def _GetPipelineDictRoutines():
    func_dict = {}

    # nltk:
    func_dict[kNltkSentencePunktTokenize]       = tpnltk.SentencePunktTokenize
    func_dict[kNltkSentenceTokenize]            = tpnltk.SentenceTokenize
    func_dict[kNltkWordTokenize]                = tpnltk.WordTokenization
    func_dict[kNltkLowerCaseTokens]             = LowerCaseTokens
    func_dict[kNltkPosTagging]                  = tpnltk.PosTagging
    func_dict[kNltkRemoveRareWords]             = tpnltk.RemoveRareWords
    func_dict[kNltkRemoveStopWords]             = tpnltk.RemoveStopWordsFromTokens
    func_dict[kNltkRemovePunktuationTokens]     = RemovePunktuationTokens
    #func_dict[kNltkRemovePunktuations] = tpnltk.RemovePunktuations

    func_dict['nltk.lemmatizer'] = True

    func_dict['nltk.RemovePunctuation'] = True
    func_dict['nltk.SnowballStemmer'] = True
    func_dict['nltk.PorterStemmer'] = True

    #
    func_dict['nltk.bow'] = True
    func_dict['nltk.Tfidf'] = True
    #
    func_dict['nltk.skip-gram'] = True
    func_dict['nltk.cbow'] = True

    #
    func_dict['nltk.word2vec'] = True
    func_dict['nltk.glove'] = True


    # RegEx routines:
    func_dict[kReRemoveHtlmTags]                    = tpregex.clean_html_tags
    func_dict[kReRemoveUrls]                        = tpregex.remove_urls
    func_dict[kReNormalizePunctuationAndLower]      = tpregex.normalize_words_punctuation_and_lower_case
    func_dict[kReRemoveTextPunctuation]             = tpregex.remove_punct_leave_only_en_letters
    func_dict[kReRemoveWhiteSpacesDuplicates]       = tpregex.remove_whitespaces_duplicates



    # Sklearn routines:

    return func_dict



# nltk
class TextPreprocessorWrapper:
    def __init__(self):
        return

    #def GetFunction(self, func_name):
    #    callbacks = _GetPipelineDictRoutines()

    #    func = callbacks.get(func_name)
    #    if func is not None:
    #        return func
    #    logger.error("Function ", func_name, " not found")
    #    return None

    def RunFunction(self, func_name, in_params):
        callbacks = _GetPipelineDictRoutines()

        func = callbacks.get(func_name)
        if func is not None:

            # functiojn overload for pd.Series
            if isinstance(in_params, pd.Series):
                result = in_params.apply(func)
                return result

            # else - apply raw
            result = func(in_params)
            return result

        logger.error("Function ", func_name, " not found")
        return None
