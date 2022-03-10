from MHFA.data import get_tweets_and_reddit
from MHFA.preprocessing import clean_text
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


class Bayes():
    def __init__(self, X, y):
        """
        X = Sequence of texts
        y = Sequence of labels. 0 (not_depressed) and 1 (depressed)
        """
        self.X = X
        self.y = y
        self.pipeline = None


    def set_pipeline(self):
        """ Defines the pipeline as a class atribute """
        self.pipeline = Pipeline([
                        ('tfidf', TfidfVectorizer()),
                        ('nb', MultinomialNB())
                            ])

    def run(self):
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """ Returns a classification report of the trained model """
        y_pred = self.pipeline.predict(X_test)
        print(classification_report(y_test, y_pred))



if __name__ == "__main__":
    # Get the data
    data = get_tweets_and_reddit()
    # Clean the data
    data.loc[:,'text'] = clean_text(data['text'])
    # Holdout
    X = data['text'].values
    y= data['label'].values
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
    # Instanciate the model
    bayes_model = Bayes(X=X_train, y=y_train)
    # Set the pipeline and fit the model
    bayes_model.run()
    bayes_model.evaluate(X_test, y_test)
