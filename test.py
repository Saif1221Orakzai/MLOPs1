import unittest
import json
from app import app

class TestHousePricePredictionAPI(unittest.TestCase):
    
    def test_predict(self):
        tester = app.test_client(self)
        data = {
            "BEDROOMS": 3,
            "BATHROOMS": 2,
            "GARAGE": 1,
            "LAND_AREA": 450,
            "FLOOR_AREA": 120,
            "BUILD_YEAR": 1995,
            "CBD_DIST": 12000,
            "NEAREST_STN_DIST": 1500,
            "LATITUDE": -32.1159,
            "LONGITUDE": 115.8425,
            "NEAREST_SCH_DIST": 0.8,
            "NEAREST_SCH_RANK": 100
        }

        response = tester.post('/predict', data=json.dumps(data), content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.data)
        self.assertIn('predicted_price', response_data)

    def test_missing_field(self):
        tester = app.test_client(self)
        
        data = {
            "BATHROOMS": 2,
            "GARAGE": 1,
            "LAND_AREA": 450,
            "FLOOR_AREA": 120,
            "BUILD_YEAR": 1995,
            "CBD_DIST": 12000,
            "NEAREST_STN_DIST": 1500,
            "LATITUDE": -32.1159,
            "LONGITUDE": 115.8425,
            "NEAREST_SCH_DIST": 0.8,
            "NEAREST_SCH_RANK": 100
        }

        response = tester.post('/predict', data=json.dumps(data), content_type='application/json')
        self.assertEqual(response.status_code, 400)

if __name__ == '__main__':
    unittest.main()