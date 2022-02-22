from this import d
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder

from TaxiFareModel.data import get_data, clean_data, hold_out, get_model
from TaxiFareModel.utils import compute_rmse


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        # create distance pipeline
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())])

        # create time pipeline
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))])

        """defines the pipeline as a class attribute"""
        # create preprocessing pipeline
        self.preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, [
             "pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")

        # display preprocessing pipeline
        return self.preproc_pipe

    def run(self):
        """set and train the pipeline"""

        self.preproc_pipe.fit(self.X_train, self.y_train)

        # compute y_pred on the test set
        self.y_pred = self.preproc_pipe.predict(self.X_test)
        return self.y_pred

    def evaluate(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test
        """evaluates the pipeline on df_test and return the RMSE"""
        self.y_pred = self.preproc_pipe.predict(self.X_test)
        rmse = compute_rmse(self.y_pred, self.y_test)
        return rmse


if __name__ == "__main__":
    # get data
    df = get_data()

    # clean data
    df = clean_data(df)

    # hold out
    X_train, X_test, y_train, y_test = hold_out(df)

    # create_model
    model = get_model()

    # trainer.run()

    # evaluate
    Trainer.evaluate(X_test, y_test, pipeline)
