import sagemaker
from sagemaker import get_execution_role
from sagemaker.sklearn.estimator import SKLearn

role = get_execution_role()
session = sagemaker.Session()

# Define SKLearn estimator
sklearn_estimator = SKLearn(entry_point='train.py',
                            role=role,
                            instance_type='ml.m4.xlarge',
                            framework_version='0.23-1')

# Train the model
sklearn_estimator.fit({'train': 's3://your-bucket/path/to/train_data.csv'})
predictor = sklearn_estimator.deploy(instance_type='ml.m4.xlarge', initial_instance_count=1)
result = predictor.predict(data)
