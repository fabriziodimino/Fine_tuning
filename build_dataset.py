import pandas as pd
import openai
import os
import asyncio
from dotenv import load_dotenv
from docling.document_converter import DocumentConverter

# Load environment variables
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY", "")
client = openai.AsyncOpenAI()

# Load CSV file
df = pd.read_csv("df.csv")

# Parsing
def parse(pdf_path):
    """Reads and extracts text from a PDF using DocumentConverter."""
    converter = DocumentConverter()
    extracted_text = converter.convert(pdf_path)
    return extracted_text.document.export_to_markdown()

pdf_document = "airbnb.pdf"
file = parse(pdf_document)

# Generate Responses
async def output(query):
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Answer user questions using the information available in this {file}."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": query
                    }
                ]
            }
        ],
        response_format={
            "type": "text"
        },
        temperature=0,
        max_completion_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    
    return response.choices[0].message.content

async def process_queries():
    tasks = [output(query) for query in df['query']]
    responses = await asyncio.gather(*tasks)
    return responses

# Run the async process
responses = asyncio.run(process_queries())
df['response'] = responses
df.to_csv('new_df.csv', index=False)