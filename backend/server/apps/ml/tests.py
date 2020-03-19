from django.test import TestCase
from apps.ml.registry import MLRegistry
from apps.ml.risk_classifier.random_forest import RandomForestClassifier

class MLTests(TestCase):
    def test_rf_algorithm(self):
        input_data = {
            "gender": "f",
            "age": "B",
            "bmi": "overweight",
            "heredity": "n",
            "calorie": "y",
            "sleep": "regular",
            "bp": "normal",
            "smoke": "y",
            "alcohol": "y",
            "mental": "n",
            "physical": "n",
            "skin": "y",
            "pcos": "y"
        }
        my_alg = RandomForestClassifier()
        response = my_alg.compute_prediction(input_data)
        #self.assertEqual('OK', response['status'])
        self.assertTrue('label' in response)
        self.assertEqual('h', response['label'])
        self.assertEqual('m', response['label'])
        self.assertEqual('l', response['label'])

    def test_registry(self):
        registry = MLRegistry()
        self.assertEqual(len(registry.endpoints), 0)
        endpoint_name = "risk_classifier"
        algorithm_object = RandomForestClassifier()
        algorithm_name = "random forest"
        algorithm_status = "production"
        algorithm_version = "0.0.1"
        algorithm_owner = "Piotr"
        algorithm_description = "Random Forest with simple pre- and post-processing"
        algorithm_code = inspect.getsource(RandomForestClassifier)
        # add to registry
        registry.add_algorithm(endpoint_name, algorithm_object, algorithm_name,
                    algorithm_status, algorithm_version, algorithm_owner,
                    algorithm_description, algorithm_code)
        # there should be one endpoint available
        self.assertEqual(len(registry.endpoints), 1)
