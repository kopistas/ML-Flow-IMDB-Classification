import requests
import json
import sys
import argparse

def query_api(text, batch=False, url="http://localhost:5000"):
    """Query the sentiment API with text input"""
    
    if batch:
        if isinstance(text, str):
            texts = text.strip().split('\n')
        else:
            texts = text
        data = {"text": texts}
    else:
        data = {"text": text}
    
    try:
        response = requests.post(f"{url}/predict", json=data)
        
        if response.status_code == 200:
            result = response.json()
            print(json.dumps(result, indent=2))
            return result
        else:
            print(f"Error: API returned status code {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Query the IMDB sentiment classification API")
    parser.add_argument("--text", "-t", help="Text to classify")
    parser.add_argument("--file", "-f", help="File containing text to classify (one per line for batch)")
    parser.add_argument("--batch", "-b", action="store_true", help="Process as batch (multiple texts)")
    parser.add_argument("--url", "-u", default="http://localhost:5000", help="API URL")
    
    args = parser.parse_args()
    
    if args.text:
        text = args.text
    elif args.file:
        with open(args.file, 'r') as f:
            text = f.read()
    else:
        print("Please provide text using --text or --file")
        return
    
    query_api(text, batch=args.batch, url=args.url)

if __name__ == "__main__":
    main() 