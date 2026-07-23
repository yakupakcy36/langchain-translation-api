from dotenv import load_dotenv
from langchain.schema import chat
from langchain_core.messages import HumanMessage,SystemMessage
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from fastapi import FastAPI
from langserve import add_routes
load_dotenv()

model = ChatGroq( model="llama-3.1-8b-instant",
    temperature=0.1)


system_prompt="Translate the following into {language}"
prompt_template= ChatPromptTemplate.from_messages([("system",system_prompt),("user","{text}")])
parser=StrOutputParser()

chain= prompt_template|model|parser


app= FastAPI(title="SimpleMessage API",version="1.0",description="SimpleMessage API")
add_routes(app,chain,path="/chain")


if __name__ == '__main__':
    import uvicorn # fast api ccalıstırmak icin
    uvicorn.run(app, host="0.0.0.0", port=8000)
