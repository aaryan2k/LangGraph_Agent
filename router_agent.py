from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class ExpertRouter:
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=0.0,
            api_key=api_key
        )

        self.output_parser = StrOutputParser()

        self.nba_prompt  = ChatPromptTemplate.from_messages( 
            [
                ("system", "You are an expert on the National Basketball Association (NBA). You can answer questions about basketball, players, teams, and statistics. If you can't answer, say 'I don't know'."),
                ("human", "{input}"),
            ]
        )
        self.football_prompt  = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an expert on the National Football League (NFL). You can answer questions about football, players, teams, and statistics. If you can't answer, say 'I don't know'."),
                ("human", "{input}"),
            ]
        )

        self.tennis_prompt  = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an expert on tennis. You can answer questions about tennis, players, and statistics. Make sure to be knowledgeable about the ATP tour and Grand Slam Events. If you can't answer, say 'I don't know'."),
                ("human", "{input}"),
            ]
        )

        self.soccer_prompt  = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an expert on soccer. You can answer questions about soccer, players, teams, and statistics. Make sure to be knowledgeable about international tournaments and club leagues. If you can't answer, say 'I don't know'."),
                ("human", "{input}"),
            ]
        )

        self.f1_prompt  = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an expert on Formula 1 (F1). You can answer questions about F1, players, teams, and statistics. If you can't answer, say 'I don't know'."),
                ("human", "{input}"),
            ]
        )

        self.trainer_prompt  = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a personal trainer for any athlete. You can give advice on how to train, what drills to do, and nutritional advice. If you can't answer, say 'I don't know'."),
                ("human", "{input}"),
            ]
        )

        self.router_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a router. If the user's question is about the NBA, respond with only the word 'basketball'. "
                       "If it's about the NFL, respond with only the word 'football'. "
                       "If it's about tennis , respond with only the word 'tennis'. "
                       "If it's about soccer, respond with only the word 'soccer'. "
                       "If it's about F1, respond with only the word 'f1'. "
                       "If it's about training or nutrition for athletes, respond with only the word 'trainer'. "
                       "If it's about anything else, respond with only the word 'reject'."),
            ("human", "{input}"),
        ])


        self.nba_chain = self.nba_prompt | self.llm | self.output_parser
        self.football_chain = self.football_prompt | self.llm | self.output_parser
        self.tennis_chain = self.tennis_prompt | self.llm | self.output_parser
        self.soccer_chain = self.soccer_prompt | self.llm | self.output_parser
        self.f1_chain = self.f1_prompt | self.llm | self.output_parser
        self.trainer_chain = self.trainer_prompt | self.llm | self.output_parser
        self.router_chain = self.router_prompt | self.llm | self.output_parser

    def route_query(self, user_input: str) -> str:
            route = self.router_chain.invoke({"input": user_input}).strip().lower()
            print(f"Routing decision: {route}")
            if "basketball" in route:
                return self.nba_chain.invoke({"input": user_input})
            elif "football" in route:
                return self.football_chain.invoke({"input": user_input})
            elif "tennis" in route:
                return self.tennis_chain.invoke({"input": user_input})
            elif "soccer" in route:
                return self.soccer_chain.invoke({"input": user_input})
            elif "f1" in route:
                return self.f1_chain.invoke({"input": user_input})
            elif "trainer" in route:
                return self.trainer_chain.invoke({"input": user_input})
            else:
                return "🚫 Sorry, I can only answer questions about sports."