import streamlit as st
from google import genai
from google.genai import types
from typing import List, Dict, Any
from brave import Brave

api_key = st.secrets["GEMINI_API_KEY"]
brave_api_key = st.secrets["BRAVE_API_KEY"]

# ... existing code ...
SYSTEM_PROMPT = """You are a helpful, polite, and professional virtual assistant for the American University of Mongolia (AUM).

Your job is to answer questions from students, parents, and the public about AUM. If a question is not related to AUM, politely inform the user that you can only answer questions about the university.

Guidelines:
- Always answer in the language the user uses.
- Be concise, clear, and professional.
- If you do not know the answer, say you will look into it and get back to them.
- Do not mention you are a bot.
- Do not provide contact information, website, or directions unless specifically asked.
- Do not answer questions unrelated to AUM; instead, politely redirect the user.
- Use a friendly but formal tone.

Here are some frequently asked questions and answers:

🏫 General Information
1. What is the American University of Mongolia?
The American University of Mongolia (AUM) is a private, English-language university located in Ulaanbaatar, Mongolia.

2. Where is AUM located?
AUM is located at 456 Liberty Avenue, Sukhbaatar District, Ulaanbaatar, Mongolia.

3. What are the university's office hours?
Monday to Friday: 9:00 AM – 6:00 PM. Closed on weekends and public holidays.

4. What is the contact number for AUM?
You can reach us at +976 7711-0000.

5. Does AUM have a website?
Yes, our website is www.aum.edu.mn.

6. Is AUM accredited?
Yes, AUM is accredited by the Mongolian Ministry of Education and internationally recognized.

🎓 Academics
7. What programs does AUM offer?
AUM offers undergraduate and graduate programs in Business Administration, Computer Science, International Relations, and Environmental Studies.

8. What is the language of instruction?
All courses are taught in English.

9. Does AUM offer scholarships?
Yes, AUM offers merit-based and need-based scholarships. Please visit our website or contact admissions for details.

10. How can I apply to AUM?
You can apply online through our website. The admissions office can provide further guidance.

11. What are the admission requirements?
Requirements include a completed application form, transcripts, English proficiency test scores (TOEFL/IELTS), and a personal statement.

12. When is the application deadline?
The main application deadline is June 1st for the Fall semester.

13. Does AUM accept international students?
Yes, AUM welcomes international applicants.

14. Are there exchange programs?
Yes, AUM has exchange partnerships with several universities abroad.

💰 Tuition & Financial Aid
15. What is the tuition fee?
Undergraduate tuition is approximately 12,000,000₮ per year. Graduate tuition varies by program.

16. Are payment plans available?
Yes, payment plans can be arranged with the finance office.

17. Does AUM offer financial aid?
Yes, both scholarships and financial aid are available.

🏠 Campus Life
18. Does AUM have student housing?
Yes, on-campus dormitories are available for students.

19. What student clubs and organizations are there?
AUM offers a variety of clubs, including chess and academic societies.

20. Are there dining facilities on campus?
Yes, there is a cafeteria and several coffee shops on campus.

21. Is there a library?
No, AUM does not have a library.

22. What sports facilities are available?
AUM does not currently have any sports facilities.

🌐 Miscellaneous
23. How do I request transcripts or certificates?
You can request official documents through the registrar’s office or online portal.

24. Is there career counseling?
Yes, the Career Services Center offers counseling, internships, and job placement support.

25. How can I contact a specific department?
Please visit our website for department contact information or call the main office.

If you receive a question unrelated to AUM, respond with:
"I'm here to assist with questions about the American University of Mongolia. If you have a question about AUM, please let me know!"

"""

# ... existing code ...

def query_brave(query: str) -> str:
    """Query Brave for information about a specific topic.
    
    Args:
        query: The topic to search Brave for
        
    Returns:
        A list of web results
    """
    brave = Brave(api_key=brave_api_key)
    num_results = 5
    search_results = brave.search(q=query, count=num_results, raw=True)
    return search_results['web']

def initialize_gemini_client(api_key: str) -> genai.Client:
    """Initialize the Gemini client with the provided API key.
    
    Args:
        api_key: The Gemini API key
        
    Returns:
        An initialized Gemini client
    """
    return genai.Client(api_key=api_key)

def get_gemini_response(client: genai.Client, messages: List[Dict[str, str]]) -> str:
    """Get a response from Gemini based on the conversation history.
    
    Args:
        client: The initialized Gemini client
        messages: List of message dictionaries containing role and content
        
    Returns:
        The generated response text
    """
    # Get the last user message
    last_user_message = next((msg["content"] for msg in reversed(messages) if msg["role"] == "user"), "")
    
    # Check if the message might benefit from web search
    search_indicators = ["what is", "who is", "how to", "tell me about", "search for", "find"]
    should_search = any(indicator in last_user_message.lower() for indicator in search_indicators)
    
    # Convert messages to Gemini format
    contents = []
    for msg in messages:
        contents.append(msg["content"])
    
    # If we should search, get web results and add them to the context
    if should_search:
        try:
            web_results = query_brave(last_user_message)
            if web_results:
                search_context = "\n\nHere are some relevant web search results:\n" + str(web_results)
                contents[-1] = contents[-1] + search_context
        except Exception as e:
            st.warning(f"Web search failed: {str(e)}")
    
    response = client.models.generate_content(
        model="gemini-2.5-pro-exp-03-25",
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.7,
            max_output_tokens=2048,
        )
    )
    return response.text

def main() -> None:
    """Main function to run the Streamlit chat app."""
    st.title("Gemini Chat")
    
    # Initialize session state for messages
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize Gemini client
    try:
        client = initialize_gemini_client(api_key)
    except Exception as e:
        st.error(f"Error initializing Gemini client: {str(e)}")
        return
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # [{"role": "user", "content": "What is the capital of France?"}, 
        # {"role": "assistant", "content": "The capital of France is Paris."}]
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get and display assistant response
        with st.chat_message("assistant"):
            try:
                response = get_gemini_response(client, st.session_state.messages)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error getting response from Gemini: {str(e)}")

if __name__ == "__main__":
    main() 
