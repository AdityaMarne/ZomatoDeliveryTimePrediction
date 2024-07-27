import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Data Transformation initiated")
            # Define which columns should be ordinal-encoded and which should be scaled
            numerical_cols = [
                'Delivery_person_Age', 'Delivery_person_Ratings', 'Restaurant_latitude',
                'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude', 
                'multiple_deliveries', 'Vehicle_condition', 'distance'
            ]
            Festival_categories = ['No', 'Yes']
            Road_traffic_density_categories = ['Low', 'Medium', 'High', 'Jam']
            Type_of_vehicle_categories = ['motorcycle', 'scooter', 'electric_scooter', 'bicycle']
            Types_of_City = ['Metropolitian', 'Urban', 'Semi-Urban']
            Type_of_Weather_Codition = ['Fog', 'Stormy', 'Sandstorms', 'Windy', 'Cloudy', 'Sunny']

            # Define the preprocessing for each type of feature
            ordinal_encoded_features = ['Festival', 'Road_traffic_density', 'City']
            one_hot_encoded_features = ['Type_of_vehicle', 'Weather_conditions']
            ordinal_categories = [Festival_categories, Road_traffic_density_categories, Types_of_City]

            logging.info("Pipeline Initiated")

            # Numerical Pipeline
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])

            # Ordinal Encoded Pipeline
            ordinal_encoded_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ("Ordinal_encoder", OrdinalEncoder(categories=ordinal_categories)),
                ('scaler', StandardScaler())
            ])

            # One-Hot Encoded Pipeline
            one_hot_encoded_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder())
            ])

            # Combined Preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_cols),
                    ('label_encoded_pipeline', ordinal_encoded_pipeline, ordinal_encoded_features),
                    ('one_hot_encoded_pipeline', one_hot_encoded_pipeline, one_hot_encoded_features)
                ])
            
            logging.info('Pipeline Completed')
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initaite_data_transformation(self, train_path, test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')
            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'Time_taken (min)'
            drop_columns = [target_column_name, 'id']

            # Check if the columns exist in the DataFrame
            existing_columns_train = [col for col in drop_columns if col in train_df.columns]
            existing_columns_test = [col for col in drop_columns if col in test_df.columns]

            input_feature_train_df = train_df.drop(columns=existing_columns_train, axis=1)
            target_feature_train_df = train_df[target_column_name] if target_column_name in train_df.columns else None

            input_feature_test_df = test_df.drop(columns=existing_columns_test, axis=1)
            target_feature_test_df = test_df[target_column_name] if target_column_name in test_df.columns else None

            # Transforming using preprocessor obj
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)] if target_feature_train_df is not None else input_feature_train_arr
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)] if target_feature_test_df is not None else input_feature_test_arr

            # Logging the head of transformed datasets
            logging.info(f'Transformed Train Data (Head): \n{pd.DataFrame(train_arr).head().to_string()}')
            logging.info(f'Transformed Test Data (Head): \n{pd.DataFrame(test_arr).head().to_string()}')

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            logging.info("Exception occurred in the initiate_data_transformation")
            raise CustomException(e, sys)
