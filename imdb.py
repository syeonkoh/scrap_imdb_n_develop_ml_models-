import requests as requests
from bs4 import BeautifulSoup
from save import save_to_file

imdb = requests.get("https://www.imdb.com/search/title/?genres=drama&start=1&explore=title_type,genres&ref_=adv_nxt")
imdb_soup = BeautifulSoup(imdb.text, 'html.parser')

movies = imdb_soup.find_all("div", {"class": "lister-item-content"})


def extract_movie(html):
    ititle = html.find("h3", {"class":"lister-item-header"})
            
    #title
    title=ititle.find("a").string
            
    #titleid
    titleid = ititle.find("a")["href"].lstrip('/title/').rstrip('/')

    return {'title': title, 'id': titleid, 'reviewlink': 'https://www.imdb.com/title/tt'+titleid+'/reviews?ref_=tt_urv'}
    
def extract_imdb_pages(last):
    
    movie_data = []

    for i in range(0,last):
        print('i : ', i)
        rank = i*50
        limit = rank+1
        toend = limit+49
        print("Movies from ", limit , "to ", toend)
        url = f"https://www.imdb.com/search/title/?genres=drama&start={limit}&explore=title_type,genres&ref_=adv_nxt"
        print('url : ', url)

        imdb = requests.get(url)
        imdb_soup = BeautifulSoup(imdb.text, 'html.parser')
        movies = imdb_soup.find_all("div", {"class": "lister-item-content"})

        for movie in movies:
            m = extract_movie(movie)
            movie_data.append(m)
    
    save_to_file(movie_data)

def extract_imdb_pages2(limit):
    
    movie_data = []
    print('i : ', limit)
    toend = limit+49
    print("Movies from ", limit , "to ", toend)
    url = f"https://www.imdb.com/search/title/?genres=drama&start={limit}&explore=title_type,genres&ref_=adv_nxt"
    print('url : ', url)
    imdb = requests.get(url)
    imdb_soup = BeautifulSoup(imdb.text, 'html.parser')
    movies = imdb_soup.find_all("div", {"class": "lister-item-content"})

    for movie in movies:
        m = extract_movie(movie)
        movie_data.append(m)
        
    save_to_file(movie_data)


            
    
            
    


    
    
            
        

