#!/usr/bin/env python3
"""Quick API verification script for Gemini 2.5 Flash Lite"""

import os
import sys

# Load .env file first
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print("❌ ERROR: python-dotenv package not installed")
    print("\nTo fix:")
    print("  pip install -r requirements.txt")
    sys.exit(1)

# Check for API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ ERROR: GEMINI_API_KEY environment variable not set")
    print("\nTo fix:")
    print("1. Get API key from: https://aistudio.google.com/app/apikey")
    print("2. Create .env file: cp .env.example .env")
    print("3. Add your key to .env: GEMINI_API_KEY=your_key_here")
    print("4. Run: python test_api.py")
    sys.exit(1)

try:
    from google import genai
except ImportError:
    print("❌ ERROR: google-genai package not installed")
    print("\nTo fix:")
    print("  pip install -r requirements.txt")
    sys.exit(1)

print("🔍 Testing Gemini API connection...")
print(f"   API Key: {api_key[:20]}...{api_key[-4:]}")

try:
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite", contents="Say 'API works!' if you can read this."
    )

    print("✅ SUCCESS: API connection verified!")
    print(f"   Response: {response.text.strip()}")

    # Check token usage if available
    if hasattr(response, "usage_metadata"):
        print(
            f"   Tokens: {response.usage_metadata.prompt_token_count} in, "
            f"{response.usage_metadata.candidates_token_count} out"
        )

    print("\n✓ Ready to proceed with LLM integration")

except Exception as e:
    print("❌ ERROR: API test failed")
    print(f"   {type(e).__name__}: {e}")
    print("\nPlease check:")
    print("  - API key is valid (https://aistudio.google.com/app/apikey)")
    print("  - Internet connection is working")
    print("  - No firewall blocking Google AI Studio")
    sys.exit(1)
