from config import load_gemini

def main():
    """
    Initial connectivity test using Gemini.
    """
    model = load_gemini()
    response = model.generate_content(
        "Hello! This is a test message. Please respond with a short confirmation."
    )
    print("\nGemini Test Response:\n", response.text)


if __name__ == "__main__":
    main()
