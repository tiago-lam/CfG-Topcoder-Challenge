# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import numpy as np
import sklearn
import sys
import pickle
import warnings

warnings.filterwarnings('ignore')

## Pre-processing
def data_frame_for_this_category(data_frame, category):
    # turn into string
    data_frame[category] = data_frame[category].astype(str)

    from sklearn.preprocessing import OneHotEncoder
    category_column = data_frame[[category]]
    one_hot_encoder = OneHotEncoder()
    sparse_array = one_hot_encoder.fit_transform(category_column)

    columns = []
    for i in range(len(one_hot_encoder.categories_[0])):
        columns.append(category + "_" + one_hot_encoder.categories_[0][i])

    data_frame_category = pd.DataFrame(sparse_array.toarray(), columns=columns)

    return data_frame_category


def create_sparse_dataset(data_training, categorical_variables):
    pre_processed_data_training = data_training
    for cat_var in categorical_variables:
        data_frame_for_this = data_frame_for_this_category(pre_processed_data_training, cat_var)

        for var_column in data_frame_for_this.columns:
            col = list(data_frame_for_this[var_column])
            pre_processed_data_training[var_column] = col

    for cat_var in categorical_variables:
        if cat_var in set(pre_processed_data_training.columns):
            pre_processed_data_training = pre_processed_data_training.drop(cat_var, axis=1)

    return pre_processed_data_training


def normalize_column(data, cName):
    data[cName] = ((data[cName] - data[cName].min()) / (data[cName].max() - data[cName].min()))
    return data


def encoding_caterigal_variables(categorical_variables, data_training):
    pre_processed = create_sparse_dataset(data_training=data_training, categorical_variables=categorical_variables)
    return pre_processed


def normalizing_variables(variables_to_normalize, data_training):
    for var in variables_to_normalize:
        data_training = normalize_column(data_training, var)
    return data_training


def main():
    if len(sys.argv) < 2 or len(sys.argv[1]) == 0:
        print("Training input file is missing.")
        return 1

    if len(sys.argv) < 3 or len(sys.argv[2]) == 0:
        print("Training output file is missing.")
        return 1

    print('Training started.')

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    train_data = pd.read_csv(input_file)
    train_data = train_data.drop_duplicates()

    non_preprocess = train_data
    y_n = np.array(non_preprocess['cancer'])
    features = list(non_preprocess.columns)
    features.remove('cancer')
    features.remove('id')
    features.remove('count')
    X_n = np.array(non_preprocess[['density', 'race', 'agefirst', 'brstproc', 'invasive']])

    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, n_jobs=-1,
                   random_state=42, solver='saga', warm_start=True)
    model.fit(X_n, y_n)

    with open(output_file, "wb") as file:
        pickle.dump(model, file)

    print('Training finished.')

    return 0


if __name__ == "__main__":
    main()