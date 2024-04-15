'''
    Write a Python program that matches a word containing 'z', not the start or end of the word
    Write a Python program to remove leading zeros from an IP address
    Write a Python program to find the occurrence and position of substrings within a string (використайте атрбути знайденої групи)
    Write a Python program to convert a date of yyyy-mm-dd format to dd-mm-yyyy format
    Write a Python program to find all three, four, and five character words in a string
    Write a Python program to convert a camel-case string to a snake-case string
    Write a Python program to find all adverbs and their positions in a given sentence
    Write a Python program to concatenate the consecutive numbers in a given string.

Original string:
Enter at 1 20 Kearny Street. The security desk can direct you to floor 1 6. Please have your identification ready.
After concatenating the consecutive numbers in the said string:
Enter at 120 Kearny Street. The security desk can direct you to floor 16. Please have your identification ready.
'''

import re

#1 Python program that matches a word containing 'z', not the start or end of the word
def text_match(text):
        patterns = '\Bz\B'
        if re.search(patterns,  text):
                return True
        else:
                return False

print(text_match("The quick brown fox jumps over the lazy dog."))
print(text_match("Python Exercises."))
print(text_match("Python Exercises.z"))


#2 Write a Python program to remove leading zeros from an IP address
ip = "216.08.094.196"
string = re.sub('\.[0]*', '.', ip)
print(string)


#3 Write a Python program to find the occurrence and position of substrings within a string
text = 'Set of exercises: Python exercises, PHP exercises, C# exercises, C++ exercises'
pattern = 'exercises'
for match in re.finditer(pattern, text):
    print('Found at pos: ', match.span()[0])

#4 Write a Python program to convert a date of yyyy-mm-dd format to dd-mm-yyyy format
def change_date_format(dt):
        newFormattedDate = re.sub(r'(\d{4})-(\d{1,2})-(\d{1,2})', '\\3-\\2-\\1', dt)
        return newFormattedDate
dt1 = "2026-01-02"
print("Original date in YYY-MM-DD Format: ",dt1)
print("New date in DD-MM-YYYY Format: ",change_date_format(dt1))

#5 Write a Python program to find all three, four, and five character words in a string
text = 'This just a smaple from the  internet. Something like the quick brown fox jumps over the lazy dog.'
print(re.findall(r"\b\w{3,5}\b", text))

#6 Python program to convert a camel-case string to a snake-case string
def change_case(str):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', str)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
str = "ActuallyThisIsAlsoFromInternetGeeksForGeeks"
print(change_case(str))

#7 Python program to find all adverbs and their positions in a given sentence
text = "Clearly, he has no excuse for such behavior. Clearly, I thought."
for m in re.finditer(r"\w+ly", text):
    print('%d-%d: %s' % (m.start(), m.end(), m.group(0)))

#8 Python program to concatenate the consecutive numbers in a given string
txt = "Enter at 1 20 Kearny Street. The security desk can direct you to floor 1 6. Please have your identification ready."
print("Original string:")
print(txt)
new_txt = re.sub(r"(?<=\d)\s(?=\d)", '', txt)
print('\nAfter concatenating the consecutive numbers in the said string:')
print(new_txt)





      









