
# Links:
# https://blog.gopenai.com/text-pre-processing-pipeline-using-nltk-3c086d04ad4e

# https://www.linkedin.com/pulse/text-preprocessing-natural-language-processing-nlp-germec-phd/
# https://spotintelligence.com/2022/12/21/nltk-preprocessing-pipeline/


import logging
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from nltk.stem.snowball import SnowballStemmer

from nltk.tag import pos_tag

#nltk.download("omw-1.4")
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer
#from nltk.tokenize.api

import string

# Logger setup:
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('TextPreprocessorNltk')


# sentence tokenizatoin
def SentencePunktTokenize(text):
    """
    tokenize text to sentence
    """
    if (type(text) != type("")):
        raise Exception('Expected string argument to handle')
    pst = PunktSentenceTokenizer()
    sentences = pst.tokenize(text)
    return sentences

def SentenceTokenize(text):
    """
    tokenize text to sentence
    """
    if (type(text) != type("")):
        raise Exception('Expected string argument to handle')
    sentences = sent_tokenize(text)
    return sentences


# word tokenization
def WordTokenization(text):
    """
    tokenize text
    """
    #if (type(text) != type("")):
    #    raise Exception('Expected string argument to handle')

    tokens = word_tokenize(text)
    return tokens

#def RemovePunktuations(text):
#    punctuated_text = nltk.wordpunct_tokenize(text)
#    return punctuated_text

#
def RemoveStopWordsFromTokens(tokens):
    stop_words = set(stopwords.words('english'))
    filetered = [token for token in tokens if token.lower() not in stop_words]
    return filetered

def PosTagging(tokens):
    tagged_words = pos_tag(tokens)
    print(tagged_words)


from collections import Counter

def RemoveRareWords(tokens, threshold=5):
    word_freq = Counter(tokens)
    filtered_words = [word for word in tokens if word_freq[word] >= threshold]
    return ' '.join(filtered_words)

# 2 Stemming and Lemmatizaton

# stemming tokens
def DoTokensPorterStemmer(tokens):
    stemmer = PorterStemmer()
    singles = [stemmer.stem(plural) for plural in tokens]
    return singles

#Stemming the text
def porter_stemmer(sen):
    ps=nltk.porter.PorterStemmer()
    sen= ' '.join([ps.stem(word) for word in sen.split()])
    return sen

#Stemming and stopwords removal
def DoTokensSnowballStemmer(tokens):
    snow = SnowballStemmer(language='english')
    singles = snow.stem(tokens)
    return singles

# Lemmatizaton
def DoTokenLemmatizaton(tokens):
    if (type(tokens) != type(list())):
        raise Exception('Expected list argument to handle')
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return lemmatized_tokens

# Process tokens here:


#Removing the word 'not' from stopwords
default_stopwords = set(stopwords.words('english'))

#excluding some useful words from stop words list as we doing sentiment analysis
excluded_stop_words =  set(['against','not','don', "don't",'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't",
             'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
             'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",'shouldn', "shouldn't", 'wasn',
             "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"])

def TemExampleStemmer():
    print("Before Stemming \n tastiest \n delicious \n tasty \n tasteful\n")
    print('After Stemming')
    snow = SnowballStemmer(language='english')
    print(snow.stem('tastiest'))
    print(snow.stem('delicious'))
    print(snow.stem('tasty'))
    print(snow.stem('tasteful'))

    print("default_stopwords")
    print(default_stopwords)
    print('\n\n')
    print("excluded_stop_words")
    print(excluded_stop_words)
    print('\n\n')

    stopwords = default_stopwords - excluded_stop_words

    print("stopwords")
    print(stopwords)

#Stemming Lemmatization




#Apply function on review column
#df['review']=df['review'].apply(stemmer)

# Tokenize text using NLTK
#tokens = nltk.word_tokenize(text)
#print(tokens)




###############################################################################

'''


def _GetPipelineDictRoutines():
    func_dict = {}
    func_dict[kNltkSentencePunktTokenize] = SentencePunktTokenize
    func_dict[kNltkSentenceTokenize] = SentenceTokenize
    func_dict[kNltkWordTokenize] = WordTokenization
    func_dict[kNltkLowerCaseTokens] = LowerCaseTokens
    func_dict[kNltkPosTagging] = PosTagging
    func_dict[kNltkRemoveRareWords] = RemoveRareWords

    func_dict[kNltkRemoveStopWords] = RemoveStopWords


    func_dict[kNltkRemoveTokensPunktuations] = RemoveTokensPunktuations
    #func_dict[kNltkRemovePunktuations] = RemovePunktuations


    func_dict['nltk.lemmatizer'] = True

    func_dict['nltk.RemovePunct'] = True
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

    return func_dict

# nltk
class _TextPreprocessorNltk:
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

# Development demo
if __name__ == '__main__':
    text = """Inulinases are used for the production of high-fructose syrup and fructooligosaccharides, and are widely utilized in food and pharmaceutical industries. In this study, different carbon sources were screened for inulinase production by Aspergillus niger in shake flask fermentation. Optimum working conditions of the enzyme were determined. Additionally, some properties of produced enzyme were determined [activation (Ea)/inactivation (Eia) energies, Q10 value, inactivation rate constant (kd), half-life (t1/2), D value, Z value, enthalpy (ΔH), free energy (ΔG), and entropy (ΔS)]. Results showed that sugar beet molasses (SBM) was the best in the production of inulinase, which gave 383.73 U/mL activity at 30 °C, 200 rpm and initial pH 5.0 for 10 days with 2% (v/v) of the prepared spore solution. Optimum working conditions were 4.8 pH, 60 °C, and 10 min, which yielded 604.23 U/mL, 1.09 inulinase/sucrase ratio, and 2924.39 U/mg. Additionally, Ea and Eia of inulinase reaction were 37.30 and 112.86 kJ/mol, respectively. Beyond 60 °C, Q10 values of inulinase dropped below one. At 70 and 80 °C, t1/2 of inulinase was 33.6 and 7.2 min; therefore, inulinase is unstable at high temperatures, respectively. Additionally, t1/2, D, ΔH, ΔG values of inulinase decreased with the increase in temperature. Z values of inulinase were 7.21 °C. Negative values of ΔS showed that enzymes underwent a significant process of aggregation during denaturation. Consequently, SBM is a promising carbon source for inulinase production by A. niger. Also, this is the first report on the determination of some properties of A. niger A42 (ATCC 204,447) inulinase."""
    txtPreprocessr = _TextPreprocessorNltk()
    #settings = {'RemoveHtlmTags' : True}
    #settings = GetDefaultSettings()
    #preprocessed = txtPreprocessr.PreprocessText('This is a demo testtext!<> Visit us at Website: https://www.geeksforgeeks.org/', settings)
    #preprocessed = txtPreprocessr.RunFunction(kNltkSentenceTokenize, text)
    #logger.info(preprocessed)
    tokens = txtPreprocessr.RunFunction(kNltkWordTokenize, text)
    logger.info(tokens)

    #tokens_1 = txtPreprocessr.RunFunction(kNltkRemovePunktuations, text)

    tokens_1 = txtPreprocessr.RunFunction(kNltkRemoveTokensPunktuations, tokens)
    logger.info("")
    logger.info(tokens_1)

'''
