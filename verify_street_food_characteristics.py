import torch
import unittest
from unittest.mock import MagicMock, patch
from services.stable_food_pipeline import StableOpenCLIPClassifier, STREET_FOOD_CHARACTERISTICS, FoodGroupPrediction

class TestStreetFoodLogic(unittest.TestCase):
    @patch('services.stable_food_pipeline.open_clip')
    @patch('services.stable_food_pipeline.torch')
    def setUp(self, mock_torch, mock_open_clip):
        # Mock the model and tokenizer
        self.mock_model = MagicMock()
        self.mock_tokenizer = MagicMock()
        
        # Setup mock embeddings for characteristics
        # 4 characteristics, each with embedding of size 512
        mock_char_embeddings = torch.randn(len(STREET_FOOD_CHARACTERISTICS), 512)
        mock_char_embeddings = mock_char_embeddings / mock_char_embeddings.norm(dim=-1, keepdim=True)
        
        with patch.object(StableOpenCLIPClassifier, '_load_model'):
            with patch.object(StableOpenCLIPClassifier, '_load_food_groups'):
                with patch.object(StableOpenCLIPClassifier, '_precompute_food_group_embeddings'):
                    self.classifier = StableOpenCLIPClassifier()
                    self.classifier.model = self.mock_model
                    self.classifier.tokenizer = self.mock_tokenizer
                    self.classifier.device = "cpu"
                    self.classifier.characteristic_embeddings = mock_char_embeddings

    def test_detect_street_food_characteristics(self):
        # Create a mock image embedding (1, 512)
        image_embedding = torch.randn(1, 512)
        image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
        
        # Manually set the first characteristic embedding to match the image embedding for high similarity
        self.classifier.characteristic_embeddings.data[0] = image_embedding.data.squeeze(0)
        
        score = self.classifier.detect_street_food_characteristics(image_embedding)
        
        # The score should be near 1.0 since it matches exactly
        self.assertGreater(score, 0.9)
        self.assertLessEqual(score, 1.01)

    def test_classify_priority_boost(self):
        # Mock image encoding
        self.classifier._encode_image = MagicMock(return_value=torch.randn(1, 512))
        
        # Mock food group predictions: [Other, Other, Street Food]
        # Street food is Top-3 but not Top-2
        groups = [
            FoodGroupPrediction(name="dal", confidence=0.5, cuisine="indian"),
            FoodGroupPrediction(name="wet curry", confidence=0.3, cuisine="indian"),
            FoodGroupPrediction(name="street food", confidence=0.1, cuisine="indian")
        ]
        self.classifier._predict_food_groups = MagicMock(return_value=groups)
        
        # Mock characteristic score to be HIGH (triggering priority)
        self.classifier.detect_street_food_characteristics = MagicMock(return_value=0.25)
        
        # Mock retrieval to return something empty or mock
        self.classifier._load_dish_embeddings = MagicMock(return_value=([], torch.Tensor()))
        self.classifier._retrieve_dishes = MagicMock(return_value=[])
        
        results = self.classifier.classify(MagicMock())
        
        # Verify that "street food" was added to candidate_groups despite being 3rd
        # We can't easily check internal candidate_groups, but we check if _load_dish_embeddings was called for street food
        calls = [call[0][0] for call in self.classifier._load_dish_embeddings.call_args_list]
        # Find the dish_file associated with street food from FOOD_GROUP_CONFIG
        from services.stable_food_pipeline import FOOD_GROUP_CONFIG
        sf_file = FOOD_GROUP_CONFIG["street food"]["dish_file"]
        self.assertIn(sf_file, calls)

if __name__ == '__main__':
    unittest.main()
