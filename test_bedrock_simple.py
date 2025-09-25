#!/usr/bin/env python3
"""
Simple Bedrock Test

Test if the specific models you have access to work correctly.
"""

import boto3
import os
from dotenv import load_dotenv

def test_bedrock_models():
    """Test the specific Bedrock models you have access to"""
    print("🔍 Testing Bedrock Models...")
    print("=" * 50)
    
    load_dotenv()
    
    try:
        # Create Bedrock client
        bedrock_client = boto3.client(
            'bedrock-runtime',
            region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        )
        
        # Test Titan Embeddings V2
        print("Testing Amazon Titan Embeddings V2...")
        try:
            response = bedrock_client.invoke_model(
                modelId="amazon.titan-embed-text-v2",
                body='{"inputText": "Hello world"}',
                contentType="application/json"
            )
            print("✅ Titan Embeddings V2: WORKING")
        except Exception as e:
            print(f"❌ Titan Embeddings V2: {str(e)}")
        
        # Test Amazon Nova Lite
        print("\nTesting Amazon Nova Lite...")
        try:
            response = bedrock_client.invoke_model(
                modelId="amazon.nova-lite",
                body='{"prompt": "Hello, how are you?", "maxTokens": 50}',
                contentType="application/json"
            )
            print("✅ Amazon Nova Lite: WORKING")
        except Exception as e:
            print(f"❌ Amazon Nova Lite: {str(e)}")
        
        print("\n🎉 Both models are working! Your RAG application should work now.")
        return True
        
    except Exception as e:
        print(f"❌ Bedrock Connection Failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_bedrock_models()
