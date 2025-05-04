# creative_agent.py
import os
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

def get_credentials():
    """Retrieves Watsonx.ai credentials from environment variables."""
    

    api_key = os.getenv("WATSONX_API_KEY")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    url = "https://us-south.ml.cloud.ibm.com" # Default URL, change if needed

    if not api_key:
        raise ValueError("Missing environment variable: WATSONX_API_KEY")
    if not project_id:
        raise ValueError("Missing environment variable: WATSONX_PROJECT_ID")

    return {
        "url": url,
        "apikey": api_key,
    }

def generate_ideas(topic: str, project_id: str, credentials: dict):
    """Generates creative ideas using the Granite model."""

    # Note: Trying granite-13b-chat-v2 first. 
    # If this model isn't available in your project or region, 
    # you might need to change it to "ibm/granite-3.1-8b-instruct" 
    # or another available model ID from your watsonx.ai project.
    # --- Changing model ID based on availability error ---
    # model_id = "ibm/granite-13b-chat-v2"
    model_id = "ibm/granite-3-8b-instruct"  # Using the 8B instruct model listed as available
    # --- End change ---
    
    # Define generation parameters - adjust creativity with temperature
    parameters = {
        GenParams.DECODING_METHOD: "sample", # Use sampling for more creative output
        GenParams.MAX_NEW_TOKENS: 250,      # Limit response length
        GenParams.MIN_NEW_TOKENS: 10,
        GenParams.TEMPERATURE: 0.8,         # Higher temperature = more randomness/creativity
        GenParams.TOP_K: 50,
        GenParams.TOP_P: 1,
        GenParams.REPETITION_PENALTY: 1.05   # Slightly discourage repeating words
    }

    # Initialize the model connection
    model = Model(
        model_id=model_id,
        params=parameters,
        credentials=credentials,
        project_id=project_id
    )

    # Craft the prompt
    prompt = f"""You are a creative assistant. Brainstorm 5 unique and interesting ideas related to the following topic: {topic}

Ideas:
1.""" # Using "1." encourages a list format

    print(f"\n🧠 Thinking about '{topic}'...")

    # Generate the response
    generated_response = model.generate_text(prompt=prompt)

    # Prepend the "1." back as the model likely continued from it
    # Also handle potential leading whitespace the model might add
    full_response = "1. " + generated_response.strip() 

    return full_response

if __name__ == "__main__":
    try:
        creds = get_credentials()
        # Get project_id again specifically for the Model class
        project_id_for_model = os.getenv("WATSONX_PROJECT_ID") 

        topic = input("Enter the topic you want creative ideas for: ")

        if topic:
            ideas = generate_ideas(topic, project_id_for_model, creds)
            print("\n✨ Here are some ideas:")
            print(ideas)
        else:
            print("No topic entered.")

    except ValueError as e:
        print(f"Error: {e}")
        print("Please ensure WATSONX_API_KEY and WATSONX_PROJECT_ID environment variables are set.")
    except Exception as e:
        # Catch potential API errors from watsonx.ai
        print(f"An error occurred during API call or processing: {e}")
        print("Please check your credentials, project ID, model ID availability, and network connection.") 