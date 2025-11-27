import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

!wget https://cdn.freecodecamp.org/project-data/books/book-crossings.zip
!unzip book-crossings.zip
books_filename = 'BX-Books.csv'
ratings_filename = 'BX-Book-Ratings.csv'

df_books = pd.read_csv(
    books_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['isbn', 'title', 'author'],
    usecols=['isbn', 'title', 'author'],
    dtype={'isbn': 'str', 'title': 'str', 'author': 'str'})
df_ratings = pd.read_csv(
    ratings_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['user', 'isbn', 'rating'],
    usecols=['user', 'isbn', 'rating'],
    dtype={'user': 'int32', 'isbn': 'str', 'rating': 'float32'})

user_counts = df_ratings['user'].value_counts()
book_counts = df_ratings['isbn'].value_counts()
good_users = user_counts[user_counts >= 200].index
df_filtered_users = df_ratings[df_ratings['user'].isin(good_users)]
good_books = book_counts[book_counts >= 100].index
df_filtered_all = df_filtered_users[df_filtered_users['isbn'].isin(good_books)]
df_merged = df_filtered_all.merge(df_books, on='isbn')
df_final = df_merged.drop_duplicates(['user', 'title'])
df_pivot = df_final.pivot(index='title', columns='user', values='rating').fillna(0)
df_matrix = csr_matrix(df_pivot.values)
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(df_matrix)

def get_recommends(book=""):
    """
    Takes a book title and returns a list of 5 similar books and their distances.
    Args:
      book (str): The title of the book (must be in the dataset).
    Returns:
      list: A list containing [book_title, [list_of_recommendations]]
            The list_of_recommendations is [title, distance] pairs.
    """
    try:
        book_index = df_pivot.index.get_loc(book)
        query_vector = df_matrix[book_index]
        distances, indices = model_knn.kneighbors(query_vector, n_neighbors=6)
        recs = []
        for i in reversed(range(1, 6)):
            idx = indices[0][i]
            title = df_pivot.index[idx]
            dist = distances[0][i]
            recs.append([title, dist])
        return [book, recs]
    except KeyError:
        return [f"Error: Book '{book}' not found in the filtered dataset.", []]
    except Exception as e:
        return [f"An error occurred: {e}", []]

  books = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
print(books)
def test_book_recommendation():
  test_pass = True
  recommends = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
  if recommends[0] != "Where the Heart Is (Oprah's Book Club (Paperback))":
    test_pass = False
  recommended_books = ["I'll Be Seeing You", 'The Weight of Water', 'The Surgeon', 'I Know This Much Is True']
  recommended_books_dist = [0.8, 0.77, 0.77, 0.77]
  for i in range(2):
    if recommends[1][i][0] not in recommended_books:
      test_pass = False
    if abs(recommends[1][i][1] - recommended_books_dist[i]) >= 0.05:
      test_pass = False
  if test_pass:
    print("Test Passed!")
  else:
    print("Test Failed.")
test_book_recommendation()
