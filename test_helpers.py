from helpers import clean_text, generate_distractors_from_doc
import unittest
from app import clean_text, generate_distractors_from_doc

class TestHelpers(unittest.TestCase):
    def test_clean_text(self):
        dirty_text = "Contact: user@example.com or visit https://example.com!"
        cleaned = clean_text(dirty_text)
        self.assertNotIn("@", cleaned)
        self.assertNotIn("http", cleaned)
        print("✅ test_clean_text passed!")

if __name__ == '__main__':
    unittest.main()
