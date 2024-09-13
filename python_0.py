import os
import requests
from typing import Dict
import json
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent, AgentExecutor, AgentType, initialize_agent, load_tools
from langchain_core.tools import tool
from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langdetect import detect
from datetime import datetime, timedelta
import difflib
import streamlit as st
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from fuzzywuzzy import process
from langchain_community.utilities import GoogleSerperAPIWrapper
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

#retrieve the data trough API
# load_dotenv() # load your .env file
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
SECTORS_API_KEY = st.secrets["SECTORS_API_KEY"]
SERPER_API_KEY = st.secrets["SERPER_API_KEY"]

def get_info(url: str) -> dict:
    headers = {"Authorization": SECTORS_API_KEY}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)
    return json.dumps(data)

def get_serper(query):
    headers = {"X-API-KEY": SERPER_API_KEY,
               'Content-Type':'application/json'}
    params = {
        'q': query,
        'hl': 'en', 
        'gl': 'us', 
    }
    url = 'https://google.serper.dev/search'

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.HTTPError as err:
            raise SystemExit(err)
    return json.dumps(data)

#generative AI
@tool
def get_company_overview(stock: str, section: str) -> str:
    """
    Retrieves specific company report or information for a given stock listed on the Indonesia Stock Exchange (IDX).

    Args:
        stock (str): The stock symbol or code representing the company.
        section (str): The section of the company report to retrieve.

    Returns:
        str: A detailed summary of the requested company report section for the specified stock.
    """
    url = f"https://api.sectors.app/v1/company/report/{stock}/?sections={section}"
    return get_info(url)

@tool
def get_top_companies_by_trx_volume(start_date: str, end_date: str, top_n: int=5, generate_chart: bool = False) -> str:
    """
    Retrieves the top most traded stocks in Indonesia by transaction volume within a specified date range.

    Args:
        start_date (str): The start date of the period in 'YYYY-MM-DD' format.
        end_date (str): The end date of the period in 'YYYY-MM-DD' format.
        top_n (int, optional): The number of top stocks to retrieve. Defaults to 5.

    Returns:
        str:
            A summary of the most traded stocks based on transaction volume, including stock names and relevant details.
            You must both "top companies" and "explanation"  are processed and outputted in the final response.
    """
    requested_start_date = start_date
    requested_end_date = end_date

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    date_diff = (end_dt - start_dt).days
    data_adj = False if date_diff >= 1 else True
    explanation = ""
    if start_dt.weekday() == 5:
        start_dt += timedelta(days=2)
    elif start_dt.weekday() == 6:
        start_dt += timedelta(days=1)

    if end_dt.weekday() == 5:
        end_dt += timedelta(days=2)
    elif end_dt.weekday() == 6:
        end_dt += timedelta(days=1)

    start_date = start_dt.strftime("%Y-%m-%d")
    end_date = end_dt.strftime("%Y-%m-%d")

    if data_adj:
        explanation = (
            f"The original requested date range was from {requested_start_date} to {requested_end_date}. "
            f"However, since stock data is not available on weekends, the data was fetched from the nearest weekdays: "
            f"{start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}."
        )
    explanation = explanation if explanation else "Data fetched for the exact requested date range."

    url = f"https://api.sectors.app/v1/most-traded/?start={start_date}&end={end_date}&n_stock={top_n}"
    json_data = get_info(url)

    data = json.loads(json_data)

    total_volume = {}
    for date, companies in data.items():
        for company in companies:
            symbol = company['symbol']
            volume = company['volume']
            price = company['price']
            if symbol not in total_volume:
                total_volume[symbol] = {"volume": 0, "price": price}
            total_volume[symbol]["volume"] += volume
            total_volume[symbol]["price"] = price

    sorted_companies = sorted(total_volume.items(), key=lambda x: x[1]['volume'], reverse=True)
    top_companies = sorted_companies[:top_n]

    result = []
    for symbol, data_info in top_companies:
        company_name = None
        for date, companies in data.items():
            for company in companies:
                if company['symbol'] == symbol:
                    company_name = company['company_name']
                    break 
        
        result.append({
            "company_name": company_name,
            "symbol": symbol,
            "total_volume": data_info['volume'],
            "price": data_info['price']
        })

    response = {
        "top_companies": result
    }
    fin = json.dumps(response, indent=4)
    return (fin + explanation)

@tool 
def get_daily_trx(stock, start_date: str, end_date: str, generate_chart: bool = False, data_type: str = 'close') -> str:
    """
    Retrieves the daily transaction volume of a specific stock listed on the Indonesia Stock Exchange (IDX) 
    within a given date range.

    Args:
        stock (str): The stock symbol or code for which to retrieve transaction data (e.g, 'BBRI').
        start_date (str): The start date of the period in 'YYYY-MM-DD' format.
        end_date (str): The end date of the period in 'YYYY-MM-DD' format.

    Returns:
        str: A summary of daily transaction volumes for the specified stocks, including date-wise volume details with your analytics-minded explanations, and optionally the chart image path if requested.
    """
    if isinstance(stock, str):
        stock = [stock] 

    if not isinstance(stock, list):
        return json.dumps({"error": "The stocks parameter must be a list of stock symbols."}, indent=4)
    
    requested_start_date = start_date
    requested_end_date = end_date

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    date_diff = (end_dt - start_dt).days
    data_adj = False if date_diff >= 1 else True

    explanation = ""
    if data_adj == True & start_dt.weekday() == 5:
        start_dt += timedelta(days=2)
    elif data_adj == True & start_dt.weekday() == 6:
        start_dt += timedelta(days=1)

    if data_adj == True & end_dt.weekday() == 5:
        end_dt += timedelta(days=2)
    elif data_adj == True & end_dt.weekday() == 6:
        end_dt += timedelta(days=1)

    if data_adj:
        explanation = (
            f"The original requested date range was from {requested_start_date} to {requested_end_date}. "
            f"However, since stock data is not available on weekends, the data was fetched from the nearest weekdays: "
            f"{start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}."
        )

    start_date = start_dt.strftime("%Y-%m-%d")
    end_date = end_dt.strftime("%Y-%m-%d")

    result = {s: [] for s in stock}

    for s in stock:
        url = f"https://api.sectors.app/v1/daily/{s}/?start={start_date}&end={end_date}"
        json_data = get_info(url)
        data = json.loads(json_data)
        
    if isinstance(data, list):
        for record in data:
            result[s].append({
                "date": record.get('date'),
                "volume": record.get('volume'),
                "close": record.get('close'),
                "market_cap": record.get('market_cap')
            })

    
    chart_path = None
    if generate_chart and result:
        print('generating chart...')
        plt.figure(figsize=(10, 5))
        for s in stock:
            stock_data = result.get(s, [])
            if stock_data:
                dates = [r['date'] for r in stock_data]
                if data_type == "volume":
                    values = [r['volume'] for r in stock_data]
                    ylabel = 'Transaction Volume'
                elif data_type == "close":
                    values = [r['close'] for r in stock_data]
                    ylabel = 'Close Price'
                elif data_type == "market_cap":
                    values = [r['market_cap'] for r in stock_data]
                    ylabel = 'Market Capitalization'
                else:
                    return json.dumps({"error": "Invalid data type for chart."}, indent=4)

    response = {
        "stock": stock,
        "daily_transaction": result,
        "explanation": explanation if explanation else "Data fetched for the exact requested date range.",
        "chart_path": chart_path
    }
    return json.dumps(response, indent=4)


@tool
def get_earning_revenue(classifications: str, n_stock: int, sub_sector: str) -> str:
    """
    Retrieves information about growth indicators such as earnings or revenue within a specific sub-sector

    Args:
        classifications (str): The classification type for stock growth. Supported options are:
            - 'top_earnings_growth_gainers'
            - 'top_earnings_growth_losers'
            - 'top_revenue_growth_gainers'
            - 'top_revenue_growth_losers'
            The function will find the closest match if the input is not an exact match.
            
        n_stock (int): The number of top stocks to retrieve based on the selected classification and sub-sector.
        
        sub_sector (str): The sub-sector to filter the stocks by. Supported sub-sectors include:
            - 'banks'
            - 'alternative-energy'
            - 'apparel-luxury-goods'
            - 'automobiles-components'
            - 'basic-materials'
            - 'consumer-services'
            - 'financing-service'
            - 'food-beverage'
            - 'food-staples-retailing'
            - 'healthcare-equipment-providers'
            The function will find the closest match if the input is not an exact match.

    Returns:
        str: A summary of the top stocks in the specified sub-sector based on earnings or revenue.
    """
    valid_classifications = ['top_earnings_growth_gainers',
                             'top_earnings_growth_losers',
                             'top_revenue_growth_gainers',
                             'top_revenue_growth_losers']
    valid_sub_sector = ['banks',
                        'alternative-energy',
                        'apparel-luxury-goods',
                        'automobiles-components',
                        'basic-materials',
                        'consumer-services',
                        'financing-service',
                        'food-beverage',
                        'food-staples-retailing',
                        'healthcare-equipment-providers']
    # Find the closest match to the input classification
    closest_match_class = process.extractOne(classifications, valid_classifications)
    closest_match_sec = process.extractOne(sub_sector, valid_sub_sector)
    
    selected_classification = closest_match_class[0]
    selected_sec = closest_match_sec[0]
    url = f"https://api.sectors.app/v1/companies/top-growth/?classifications={selected_classification}&n_stock={n_stock}&sub_sector={selected_sec}"
    return get_info(url)

@tool 
def get_listing_perform(stock: str) -> str:
    """
    Retrieve information about a company's performance since its IPO (Initial Public Offering) listing date.

    This function fetches data on the stock's performance history from the IPO listing date using an external API, providing insights into how the company has performed in the market over time.

    Parameters:
    - stock (str): The stock ticker symbol of the company (e.g., 'AAPL' for Apple, 'TSLA' for Tesla).

    Returns:
    - str: A response containing information about the company's performance since its IPO, fetched from the external API.
    """
    url = f"https://api.sectors.app/v1/listing-performance/{stock}/"
    json_data = get_info(url)
    
    # Parse the JSON data
    data = json.loads(json_data)
    
    # Convert float values to percentages and integers
    if data.get('chg_7d') is not None:
        data['chg_7d'] = int(round(data['chg_7d'] * 100, 2))
    if data.get('chg_30d') is not None:
        data['chg_30d'] = int(round(data['chg_30d'] * 100, 2))
    if data.get('chg_90d') is not None:
        data['chg_90d'] = int(round(data['chg_90d'] * 100, 2))
    if data.get('chg_365d') is not None:
        data['chg_365d'] = int(round(data['chg_365d'] * 100, 2))
    
    # Return the updated data as a JSON string
    return json.dumps(data, indent=4)

@tool
def process_data_with_llm(query):
    """
    Process user queries that require general knowledge or non-financial information beyond the scope of the provided API database.

    This function serves as an interface to retrieve general information by leveraging an LLM or search engine (e.g., Serper), allowing the system to handle a wider range of user queries, such as general knowledge, industry trends, or other non-financial data.

    Parameters:
    - query (str): The user's input question or query that needs to be processed and searched.

    Returns:
    - dict: The response from the search or LLM, containing relevant information or answers to the query.
    """
    # Search using Serper
    return get_serper(query)

tools = [get_company_overview,
         get_top_companies_by_trx_volume,
         get_daily_trx,
         get_earning_revenue,
         get_listing_perform,
         process_data_with_llm
         ]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""You are an advanced AI assistant specialized in providing accurate, insightful, and context-aware responses about company financial reports and stock information with high caliber capability on interpret and analyze the data to take perfect conclusions about it.
                On other hand also you're capable to answer general knowledge questions. Follow these guidelines to ensure the highest quality output:
                1. **Accuracy & Factuality:** Always prioritize correctness. Base your responses on verified information. If you are unsure about an answer, acknowledge it and provide the best possible estimate or recommendation.
                2. **Context Awareness & Adaptability:** Consider the entire context of the query. Infer missing details logically based on standard assumptions or previously provided information. Adjust your responses based on the user’s expertise level and tone.
                3. **Date Handling:** Handle dates accurately:
                    - today's year is {datetime.today().year}
                    - If the query mentions today's date or omits a date, use {datetime.today().strftime("%Y-%m-%d")}.
                    - If real-time data is unavailable, use {(datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")}.
                    - For date ranges, infer missing dates logically.
                4. **Parameter Extraction:** Prioritize and extract key financial parameters from the query, such as stock tickers, financial ratios, and date ranges, to ensure accurate and relevant responses.
                5. **Clarity & Precision:** Provide clear, concise responses. Avoid unnecessary jargon unless required by the context.
                6. **Analysis & Insight:** Go beyond surface to high detailed-level responses. Provide accurate analysis based on factual numeric, highlight trends, and offer predictions based on the data at hand.
                7. **User Engagement:** Encourage further interaction by suggesting next steps, asking clarifying questions, or offering additional relevant information.
                8. **Ethical Considerations:** Always adhere to ethical guidelines. Avoid providing harmful, sensitive, or biased information.
                9. **Language Flexibility:** **Language Flexibility:** For queries in Indonesian, translate the financial parameters and respond in Indonesian while keeping the structure accurate.
                10. **Multi-tasking:** If a query contains multiple questions, use different tools simultaneously to retrieve the necessary information.
                11. **Precise:** 
                    - You are able to convert prompt question into parameters that claas need to so you could access the API to get an answer.
                    - docstring on each @tools classes is your guidance to get an API urls as answer source(s).
                12. If you detect the general question, you should use process_data_with_llm tools that enhanced your knowledge with google serper search engine.
                13. If you find out confussion to pick parameters from questions, you could put default parameter as null.
                14. You should aware enough with the instructions on docstring each tools. it will guide you to answer the user's question/promp/query.
                15. **Formatting:** You should give answer as a combination of paragraph and list if necessary, don't answer it as paragraph only.
                16. If your response has explanation on its JSON data, you should mention or say it on your final answer to the question.
                17. **Performing Calculations:** When calculation is required and the LLM cannot do it directly, ensure you still provide a solution by outlining the steps involved or leveraging external tools and logic to resolve the issue.
                18. **Smart Calculation Handling:** Although the AI LLM has limitations with direct calculations, you are equipped to perform necessary calculations by breaking them down into steps or using external resources.
                19. When you couldn't answer user questions, you should say "im not capable to answer your questions".
                20. your only tools are get_company_overview, get_top_companies_by_trx_volume, get_daily_trx, get_stock_info, get_listing_perform, process_data_with_llm.
                Deliver your answers as if you are the most knowledgeable expert on the subject, maintaining a user-friendly and approachable tone."""
                       
         ),
        ("human", "{input}"),
        # msg containing previous agent tool invocations 
        # and corresponding tool outputs
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

llm = ChatGroq(
    temperature=0,
    model_name="llama3-groq-70b-8192-tool-use-preview",
    groq_api_key=GROQ_API_KEY,
)

agent = create_tool_calling_agent(llm, tools, prompt)
memoryforchat=ConversationBufferMemory()
agent_executor = AgentExecutor(memory=memoryforchat, agent=agent, tools=tools, verbose=True)

# streamlit aestetatichally purposes
st.set_page_config(page_title="Chit Stack Stock Machina",
                   page_icon = "🤖",
                   layout = 'centered')
st.title("📈💲Chit Stack Stock Machina")
st.subheader("*Your **AI-powered companion for real-time financial insights and stock analysis.** Precision, intelligence, and data-driven decisions at your fingertips*")
st.sidebar.title("📈💲Chit Stack Stock Machina")
st.sidebar.subheader("Frequently Asked Questions🔥")
faq = ["What are the top 5 companies by transaction volume on the first of this month?",
                    "What are the most traded stock yesterday?",
                    "What are the top 7 most traded stocks between 6th June to 10th June this year?",
                    "What are the top 3 companies by transaction volume over the last 7 days?",
                    "Based on the closing prices of BBCA between 1st and 30th of June 2024, are we seeing an uptrend or downtrend? Try to explain why.",
                    "What is the company with the largest market cap between BBCA and BREN? For said company, retrieve the email, phone number, listing date and website for further research.",
                    "What is the performance of GOTO (symbol: GOTO) since its IPO listing?",
                    "If i had invested into GOTO vs BREN on their respective IPO listing date, which one would have given me a better return over a 90 day horizon?",
                    ]

if 'selected_question' not in st.session_state:
    st.session_state.selected_question = ""

# Display template questions as buttons in the sidebar
for question in faq:
    if st.sidebar.button(question):
        # When a button is clicked, set the question to session state
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        st.session_state.selected_question = question
        st.chat_message("user").markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})
        response = agent_executor.invoke({"input": question})
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                time.sleep(1)
                st.markdown(response['output'])
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response['output']})

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Text to Machina"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        response = agent_executor.invoke({"input": prompt})
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                time.sleep(1)
                st.markdown(response['output'])
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response['output']})

    except IndexError as e:
        st.error(f"Index Error: {e}")
        st.session_state.messages.append({"role": "assistant", "content": "Sorry, something went wrong with the list handling."})
    except Exception as e:
        st.error(f"Error invoking agent: {e}")
        st.session_state.messages.append({"role": "assistant", "content": "Sorry, something went wrong. Could you please specify the question?"})
    except requests.exceptions.HTTPError as http_err:
        st.error(f"(Status Code: {response.status_code})")
        st.session_state.messages.append({"role": "assistant", "content": "Sorry, something went wrong. Could you please specify the question?"})
                
st.session_state.selected_question = prompt

        

# queries = [query_5]

# for query in queries:
#     print("Question:", query)
#     result = agent_executor.invoke({"input": query})
#     print("Answer:", "\n", result["output"], "\n\n======\n\n")

# query = "What are the most traded stock yesterday?"
# print("Question:", query)
# result = agent_executor.invoke({"input": query})
# print("Answer:", "\n", result["output"], "\n\n======\n\n")
