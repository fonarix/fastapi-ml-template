# https://lms.ithillel.ua/groups/65ccf134073f165247467f0c/lessons/65ccf134073f165247467f22

# Read:
# https://www.kaggle.com/code/sambhavsg/nlp-basics-imdb-movie-review-sentiment
# https://www.linkedin.com/pulse/text-preprocessing-natural-language-processing-nlp-germec-phd/


import pandas as pd
import TextPreprocessorWrapper as tpp

data = pd.read_csv('../datasets/IMDB Dataset.csv')

print("")
print(data)

print("")
print(data.describe())

def DoPreprocessingDev(data):
    txt_preprocessor = tpp.TextPreprocessorWrapper()
    # Process as string
    result_series = txt_preprocessor.RunFunction(tpp.kReRemoveUrls, data['review'])
    result_series = txt_preprocessor.RunFunction(tpp.kReRemoveHtlmTags, result_series)
    # Replace "you'll" to "you will"
    result_series = txt_preprocessor.RunFunction(tpp.kReNormalizePunctuationAndLower, result_series)
    result_series = txt_preprocessor.RunFunction(tpp.kReRemoveTextPunctuation, result_series)

    # Make tokens
    result_series = txt_preprocessor.RunFunction(tpp.kNltkWordTokenize, result_series)
    result_series = txt_preprocessor.RunFunction(tpp.kNltkRemovePunktuationTokens, result_series)
    data['review_tokenized'] = result_series
    print(data.head())
    return data


def DoPreprocessingWithProcessors(data, list_of_preprocessors):
    txt_preprocessor = tpp.TextPreprocessorWrapper()

    result_series = data['review']

    for preproc in list_of_preprocessors:
        print("Running process: ", preproc)
        result_series = txt_preprocessor.RunFunction(preproc, result_series)

    data['review_tokenized'] = result_series
    return data

preproc_list = [
    # Process as string
    tpp.kReRemoveUrls,
    tpp.kReRemoveHtlmTags,
    # Replace "you'll" to "you will"
    tpp.kReNormalizePunctuationAndLower,
    # Remove rest of punctuations
    tpp.kReRemoveTextPunctuation,

    # Make tokens
    tpp.kNltkWordTokenize,
    # can be skipped, we removed punctuations from text
    tpp.kNltkRemovePunktuationTokens
]

#data = DoPreprocessingDev(data)

data = DoPreprocessingWithProcessors(data, preproc_list)

print(data.head())


print("End")




