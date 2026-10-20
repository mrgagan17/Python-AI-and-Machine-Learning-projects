import nltk
from nltk.chat.util import Chat, reflections

# Define pattern-response pairs for the customer support bot
# (r"pattern", ["response 1", "response 2"])
patterns = [
    (r"(hi|hello|hey|hola)(.*)", 
     ["Hello! I am your AI Support Assistant. How can I help you today?", "Hi there! What can I assist you with?"]),
    
    (r"my name is (.*)", 
     ["Nice to meet you, %1! How can I help you today?"]),
    
    (r"what is your name\??", 
     ["I am a rule-based AI support bot built with NLTK.", "You can call me AI Assistant."]),
    
    (r"how are you\??", 
     ["I'm functioning perfectly! How can I help you with your account or order?"]),
    
    (r"(.*)(help|support)(.*)", 
     ["I can help you with order tracking, refunds, or general inquiries. What do you need?"]),
    
    (r"(.*) order (.*)", 
     ["To check your order status, please provide your 6-digit Order ID.", "I can help with that. What is your order number?"]),
    
    (r"(.*)(refund|return)(.*)", 
     ["To process a refund or return, please ensure it is within 30 days of purchase. Would you like the link to our returns portal?"]),
    
    (r"(.*) human (.*)", 
     ["I am a bot, but I can transfer you to a human agent. Please hold on..."]),
    
    (r"quit|exit|bye|goodbye", 
     ["Goodbye! Have a great day!", "Bye! Feel free to reach out if you need more help."]),
    
    # Fallback pattern for unrecognized intents
    (r"(.*)", 
     ["I'm sorry, I didn't quite catch that. Could you rephrase your question?", "I only understand basic support queries right now. Could you ask about orders, refunds, or support?"])
]

def get_bot_response(user_input):
    # Initialize the NLTK Chat utility with our patterns and basic pronoun reflections (e.g., "I" -> "you")
    chatbot = Chat(patterns, reflections)
    
    # Generate response based on regex matching
    response = chatbot.respond(user_input)
    return response if response else "I'm having trouble understanding. Please contact support@example.com."