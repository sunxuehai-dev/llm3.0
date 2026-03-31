import sys
import os
import unittest
from fastapi.testclient import TestClient

# Add project root to sys.path to allow importing 'app'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

class TestPromptTemplates(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_get_prompt_templates(self):
        """
        Test the educational prompt templates endpoint.
        """
        response = self.client.get("/api/playground/images/prompts/templates")
        
        # 1. Check status code
        self.assertEqual(response.status_code, 200, f"Expected 200, got {response.status_code}")
        
        data = response.json()
        
        # 2. Check structure
        self.assertIn("count", data)
        self.assertIn("templates", data)
        self.assertIsInstance(data["templates"], list)
        
        # 3. Check content validity
        self.assertGreater(data["count"], 0)
        self.assertEqual(len(data["templates"]), data["count"])
        
        # 4. Check specific template fields
        first_template = data["templates"][0]
        required_fields = ["id", "category", "title", "description", "prompt", "negative_prompt"]
        for field in required_fields:
            self.assertIn(field, first_template, f"Missing field '{field}' in template")
            
        print("\n✅ Prompt Templates Test Passed!")
        print(f"Found {data['count']} templates.")
        print("Sample Template:", first_template["title"])

if __name__ == "__main__":
    unittest.main()
