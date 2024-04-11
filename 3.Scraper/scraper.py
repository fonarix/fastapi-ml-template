
#В репозиторії https://github.com/realpython/fake-jobs/tree/main знаходяться HTML файли.
#Написати скрепер, який забере корисну інформацію з файлів в папці jobs
#Спочатку аналізувати основний файл, потім переходити за посиланням і працювати з конкретним файлом

from bs4 import BeautifulSoup
import requests

URL = "https://raw.githubusercontent.com/realpython/fake-jobs/main/index.html"
page = requests.get(URL)
soup = BeautifulSoup(page.content, "html.parser")

#print(soup)

allCards = soup.findAll('div', class_='card-content');
print('Total cards: ', len(allCards));


def ParseSingleJobDetails(pageReference):
    page = requests.get(pageReference)
    soup = BeautifulSoup(page.content, "html.parser")

    content = soup.find('div', class_='content')
    if content is not None:

        location = content.find('p', id='location')
        if location is not None:
            print('location: ', location.text);

        date = content.find('p', id='date')
        if date is not None:
            print('date: ', date.text);

    return


# or fill array of objects...
for singleCard in allCards:
    title = singleCard.find('h2', class_='title is-5')
    if title is not None:
        print('title: ', title.text);

    company = singleCard.find('h3', class_='subtitle is-6 company')
    if company is not None:
        print('company: ', company.text);

    #get from card-footer
    jobDetailsLink = singleCard.find("a", href=True, string="Apply")
    if jobDetailsLink is not None:
        pageRef = jobDetailsLink['href']
        pageRef = pageRef.replace("https://github", "https://raw.githubusercontent");
        print('pageRef: ', pageRef);
        ParseSingleJobDetails(pageRef);

    










