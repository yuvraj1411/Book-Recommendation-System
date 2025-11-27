📚 Book Recommender System (K-Nearest Neighbors)

This Google Colab notebook implements a Collaborative Filtering Book Recommender System using the K-Nearest Neighbors (KNN) algorithm. It utilizes the Book-Crossing dataset to identify and suggest books that are highly similar to a user-provided title based on shared user ratings.

🔗 Colab Notebook Link: https://colab.research.google.com/drive/1-_x-VwfLnL6hqKCJ5hdwAsS0T9MY5DsM?usp=sharing

✨ Key Features
1. Collaborative Filtering: Uses item-based collaborative filtering to find similar books.
2. KNN Implementation: Employs the NearestNeighbors model from scikit-learn with Cosine Similarity as the metric.
3. Data Filtering: Implements rigorous filtering to ensure model stability and performance by only including users who have rated >= 200 books & books that have received >= 100 ratings.
4. Sparse Matrix Optimization: Converts the large, sparse user-item matrix into a Compressed Sparse Row (CSR) matrix for efficient memory usage and fast computation.

⚙️ Technical Workflow
1. Data Loading: Downloads and loads two datasets: BX-Books.csv (book details) and BX-Book-Ratings.csv (user ratings).
2. Data Preprocessing: Filters the data based on user and book count thresholds. Merges book and rating data.Creates the user-item rating matrix where rows are book titles, columns are users, and values are the        ratings.
3. Model Training: The KNN model is trained on the CSR rating matrix.
4. Recommendation Function (get_recommends): Takes a book title as input. Finds the book's vector in the matrix.Uses model_knn.kneighbors to find the 5 closest neighboring books based on their cosine                   distance. Returns the recommended book titles and their respective distances.

💻 Usage

Open the Colab link and run all cells sequentially. The final cells will automatically execute the get_recommends function for a test book and run a verification test to confirm the recommendations meet the expected results.
