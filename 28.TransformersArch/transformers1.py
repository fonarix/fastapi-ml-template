
from transformers import pipeline

classifier = pipeline(
    task = 'sentiment-analysis',
    model = 'SkolkovoInstitute/russian_toxicity_classifier')

text = ['У нас в есть убунты и текникал превью.',
    	'Как минимум два малолетних дегенерата в треде, мда.']

#result = clf(text)
result = classifier(text, top_k=None)


print("---------------------------------------")
print(result)


print("End")

