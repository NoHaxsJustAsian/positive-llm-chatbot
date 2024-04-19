import unittest
from DEMO_FINAL import text_preprocessing, pad_features, analyze_user_input, create_vocab_and_encode, load_data, model
from unittest.mock import patch
import numpy as np


class TestModel(unittest.TestCase):
    def setUp(self):
        self.df = load_data('output_from_amazon_imdb_yelp.csv')
        self.df, self.vocab_to_int, self.vocab_size = create_vocab_and_encode(self.df)
        self.sequence_length = 50

    def test_analyze_user_input_negative(self):
        result = analyze_user_input("this product is awful", model, self.vocab_to_int, self.sequence_length)
        self.assertEqual(result, 'Negative')
        result = analyze_user_input("this place sucks", model, self.vocab_to_int, self.sequence_length)
        self.assertEqual(result, 'Negative')

    def test_analyze_user_input_positive(self):
        result = analyze_user_input("I am having fun", model, self.vocab_to_int, self.sequence_length)
        self.assertEqual(result, 'Positive')
        result = analyze_user_input("I love you", model, self.vocab_to_int, self.sequence_length)
        self.assertEqual(result, 'Positive')
        
    def test_analyze_user_input_false_negative(self):
        result = analyze_user_input("I am happy", model, self.vocab_to_int, self.sequence_length)
        self.assertEqual(result, 'Positive', "Expected 'Positive' but got '{}' this is due to the fact that a lot of the data is trained on sentences which include the statement \"not happy\" in it".format(result))

    def test_analyze_user_input_false_positive(self):
        result = analyze_user_input("I like you", model, self.vocab_to_int, self.sequence_length)
        self.assertEqual(result, 'Negative', "Expected 'Negative' but got '{}' this is due to the fact that a lot of the data is trained on sentences which include the statement \"don't like\" in it".format(result))\
        
if __name__ == '__main__':
    unittest.main()
