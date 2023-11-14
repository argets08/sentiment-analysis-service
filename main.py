import logging
from datetime import datetime, timedelta
from typing import List
from termcolor import colored
from alpaca.data.historical.news import NewsClient, NewsSet
from alpaca.data.requests import NewsRequest

from fastapi import FastAPI
from transformers import pipeline
import uvicorn

app = FastAPI()
# Alpaca API credentials
API_KEY = ""
API_SECRET = ""
ALPACA_BASE_URL = 'https://data.alpaca.markets'
logging.getLogger("transformers").setLevel(logging.ERROR)
SCORE_THRESHOLD = 0.99

@app.get("/")
async def root():
    return {"message": "Sentiment Analysis Service : Online"}

def extract_alpaca_news(api_key, api_secret, symbols, start_date, end_date):
    try:
        news_client = NewsClient(api_key=api_key, secret_key=api_secret)
        news_request = NewsRequest(symbols=symbols, start_date=start_date, end_date=end_date, limit=50)
        news = news_client.get_news(news_request)
        return news
    except Exception as e:
        logging.error(f"An error occurred while extracting news: {e}")
        raise


def get_article_titles(all_news: NewsSet) -> List[str]:
    headlines_list = [article.headline for article in all_news.news]
    return headlines_list

def get_sentiment_list(article_titles):
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    revision = "af0f99b"
    classifier = pipeline('sentiment-analysis', model=model_name, revision=revision)
    sentiment_list = []
    for title in article_titles:
        sentiment = classifier(title)
        # print(sentiment)
        if (sentiment[0]['score'] >= 0.9):
            sentiment_list.append(sentiment[0])
    return sentiment_list

def print_stock_sentiment(stock, average_score, overall_sentiment, color):
    # Print the results
    print(colored(f"Stock: {stock}",'blue'))
    print(f"Average Sentiment Score: {average_score:.4f}")
    print(colored(f"Overall Sentiment: {overall_sentiment}", color))
    
@app.post("/stock_list/")
async def stock_list(list_stock: str):
    today = datetime.today().weekday()
    hours = 8 if today == 0 else 12
    print(f"Hours: {hours}")
    start_date = (datetime.today() - timedelta(hours=hours)).strftime('%Y-%m-%d')
    end_date = datetime.today().strftime('%Y-%m-%d')
    list_of_stocks = [stock.strip() for stock in list_stock.split(",")]

    stock_sentiment_results = {}

    for stock in list_of_stocks:
        pos_sentiment_score = 0
        pos_score_count = 0
        
        news = extract_alpaca_news(api_key=API_KEY, api_secret=API_SECRET, symbols=stock, start_date=start_date, end_date=end_date)
        article_titles = get_article_titles(news)
        sentiment_list_for_stock = get_sentiment_list(article_titles)

        for sentiment in sentiment_list_for_stock:
            if sentiment['label'] == 'POSITIVE':
                pos_sentiment_score += sentiment['score']
                pos_score_count += 1
        
        if pos_score_count != 0:
            average_score = pos_sentiment_score / pos_score_count
            if average_score > SCORE_THRESHOLD:
                print_stock_sentiment(stock, average_score, 'POSITIVE', 'green')
                stock_sentiment_results[stock] = {'label': 'POSITIVE', 'score': average_score}
    
    return stock_sentiment_results


@app.post("/sentiment_graph/")
async def sentiment_graph(symbol: str):
    # TODO document why this method is empty
    # create a graph of the sentiment over time and trend iteratively. Use a local small db? or just a csv file?
    pass

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)