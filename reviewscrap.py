import csv
import requests 
from bs4 import BeautifulSoup

from save import save_reviews_to_file

def extract_rating_review(html):
  try:    
    val= html.find("span",{"class": "rating-other-user-rating"}).find("span").text
    
    #print('rating    :',val)

    try:
      opinion = html.find("div", {"class": "text show-more__control"}).text
      
      #print('ooo:',opinion)

    except Exception as ex:
      opinion = 0
      
      #print('ooo:',ans)

       
  except Exception as ex:

    val = 0
    #print('rating    :', val)
    try:
      opinion = html.find("div", {"class": "text show-more__control"}).text
      
      #print('ooo:',opinion)

    except Exception as ex:
      opinion = 0
      
      #print('ooo:',ans)

  return {'rating': val, 'review': opinion}
    
def file_open():
  file = open('imdb_data.csv', 'r')
  rdr = csv.reader(file)
  header = next(rdr)
  return rdr
   

def extract_review_pages(line):
  reviewdata = []
  
 #요거를 풀해보기

  
  url = line[2]
  print("scraping movie number", url)
    
  reviews_link = requests.get(url)
  reviews_soup = BeautifulSoup(reviews_link.text, 'html.parser')
    
    
    
  reviews = reviews_soup.find_all("div", {"class": "lister-item-content"})
      
         
  for review in reviews:
    line = extract_rating_review(review)
    print('......')
    reviewdata.append(line)

      
  save_reviews_to_file(reviewdata)
  


    
    
    

    

