WEB PAGE QUESTION ANSWERING BOT

This project is a Retrieval-Augmented Generation (RAG) application built with Python. It allows users to input a website URL, processes the text content, and creates an interactive chat interface where users can ask questions about the website's content. The system uses OpenAI GPT-4o for generating answers and FAISS for efficient vector retrieval.

FEATURES

1. URL Loading: Scrapes text content dynamically from user-provided URLs.
2. Live Progress Tracking: Displays real-time status updates for scraping, splitting text, and creating embeddings.
3. Chunk Calculation: Automatically calculates and displays the number of text chunks created from the webpage.
4. Vector Search: Uses OpenAI Embeddings and FAISS to find the most relevant context for your questions.
5. Contextual Answers: Provides accurate answers based strictly on the content of the processed webpage.

PREREQUISITES

You must have Python installed on your system.
You need a valid OpenAI API Key.

INSTALLATION INSTRUCTIONS

1. Clone this repository to your local machine using following command.
git clone https://github.com/Hussain0623/WebPage_ChatAI.git
2. Install the required dependencies using following command.
pip install -r requirements.txt

3. Create a file named .env in the root directory of the project. Inside this file, add your OpenAI API key like this:
OPENAI_API_KEY=your_api_key_here

HOW TO RUN

1. Open your terminal or command prompt.
2. Navigate to the project directory.
3. Run the application using the following command:
streamlit run app.py

HOW TO USE

1. Once the app is running in your browser, enter a valid URL in the text input field.
2. Click the Process URL button.
3. Wait for the system to scrape, chunk, and index the content. You will see the status update live.
4. Once processing is complete, a success message will show the total number of chunks created.
5. Type your question in the text box below and press Enter to get an answer based on the website data.

TECHNOLOGIES USED

Streamlit: For the user interface.
LangChain: For the RAG framework and chain building.
OpenAI GPT-4o: The Large Language Model used for reasoning.
FAISS: For vector storage and similarity search.

LICENSE

This project is open source.
