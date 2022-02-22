import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def get_data(nrows=10_000):
    url = 's3://wagon-public-datasets/taxi-fare-train.csv'
    # '''returns a DataFrame with nrows from s3 bucket'''
    df = pd.read_csv(url, nrows=1_000_000)
    return df


def clean_data(df, test=False):
    df = df.dropna(how='any', axis='rows')
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    if "fare_amount" in list(df):
        df = df[df.fare_amount.between(0, 4000)]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count >= 0]
    df = df[df["pickup_latitude"].between(left=40, right=42)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
    df = df[df["dropoff_latitude"].between(left=40, right=42)]
    df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
    return df


def hold_out(df):
    # set X and y
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    df = get_data(nrows=10_000)
    df = clean_data(df)


def get_model():
    model_params = {'n_estimator': 100, 'max_depth': 1}
    model = RandomForestRegressor()
    model.set_params(**model_params)
    return model


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
