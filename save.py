import csv

def save_to_file(result):
  file = open("imdb_data.csv", mode="a+")
  writer = csv.writer(file)
  #writer.writerow(['title', 'id', 'reviewlink'])
  for val in result:
    writer.writerow(list(val.values()))
      

  return 

def insert_colname_to_file(filename, collist):
  file = open(filename, mode = "w")
  writer = csv.writer(file)
  writer.writerow(collist)


def save_reviews_to_file(reviewlist):
  file = open("reviews_ratings.csv", mode="a+")
  writer = csv.writer(file)
  #writer.writerow(['rating', 'review'])
  for val in reviewlist:
    writer.writerow(list(val.values()))
  return

