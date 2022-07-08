import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from src.data import extract_training_data


if __name__ == '__main__':
    np.random.seed(2022)
    gt_data = extract_training_data()
    
    front_dens_train, front_dens_test, side_dens_train, side_dens_test, measures_train, measures_test = \
        train_test_split(gt_data.front_densities, gt_data.side_densities, gt_data.measures.all, test_size=0.33)
    model = LinearRegression()
    
    X_train = np.expand_dims(front_dens_train, 1)
    y_train = measures_train
    X_test = np.expand_dims(front_dens_test, 1)
    y_test = measures_test
    
    reg = model.fit(X_train, y_train)
    y_predict = reg.predict(X_test)
    
    for idx in range(y_predict.shape[0]):
        for meas_idx in range(y_predict.shape[1]):
            print(f'{meas_idx}) {y_predict[idx][meas_idx] * 100.}, {y_test[idx][meas_idx] * 100.}')
    
