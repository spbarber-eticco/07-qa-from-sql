import os
from dotenv import load_dotenv, find_dotenv
import re
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Cargar variables de entorno
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

# Configurar el modelo de lenguaje
llm = ChatOpenAI(model="gpt-4o-mini")

# Configurar la base de datos
sqlite_db_path = "data/street_tree_db.sqlite"
db = SQLDatabase.from_uri(f"sqlite:///{sqlite_db_path}")

def extract_sql_query(response):
    if isinstance(response, str):
        # Eliminar "SQLQuery:" si está presente
        response = re.sub(r'^SQLQuery:\s*', '', response, flags=re.IGNORECASE)
        
        # Eliminar backticks de markdown y la palabra "sql" si están presentes
        response = re.sub(r'```sql\s*(.*?)\s*```', r'\1', response, flags=re.DOTALL | re.IGNORECASE)
        
        # Buscar una consulta SQL válida
        sql_match = re.search(r'\b(SELECT\s+.*?;)', response, re.DOTALL | re.IGNORECASE)
        if sql_match:
            return sql_match.group(1).strip()
    return response  # Si no se encuentra un patrón, devolver la respuesta original

# Crear la cadena inicial
chain = create_sql_query_chain(llm, db)

# Primera ejecución
response = chain.invoke({"question": "How many species of trees are in San Francisco?"})

print("\n----------\n")
print("How many species of trees are in San Francisco?")
print("\n----------\n")
print(response)
print("\n----------\n")

print("Query executed:")
print("\n----------\n")
sql_query = extract_sql_query(response)
print(sql_query)
print("\nResult:")
print(db.run(sql_query))

print("\n----------\n")
print("Chain prompts:")
print("\n----------\n")
chain.get_prompts()[0].pretty_print()

# Configurar herramientas y cadenas adicionales
execute_query = QuerySQLDataBaseTool(db=db)
write_query = create_sql_query_chain(llm, db)

# Segunda ejecución (con query execution included)
chain = write_query | execute_query
response = chain.invoke({"question": "List the species of trees that are present in San Francisco"})

print("\n----------\n")
print("List the species of trees that are present in San Francisco (with query execution included)")
print("\n----------\n")
extracted_query = extract_sql_query(response)
print("Extracted SQL Query:")
print(extracted_query)
print("\nQuery Result:")
print(db.run(extracted_query))
print("\n----------\n")

# Configurar prompt personalizado y cadena final
sql_prompt = PromptTemplate.from_template(
    """Given the following input question, table information, and number of results to return, generate a SQL query to answer the question.
    Return ONLY the SQL query without any additional text or formatting.

    Question: {input}
    Table Info: {table_info}
    Number of Results to Return: {top_k}

    SQL Query:"""
)

answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

chain = create_sql_query_chain(llm, db, prompt=sql_prompt)

chain = (
    RunnablePassthrough.assign(query=write_query)
    .assign(
        result=lambda x: db.run(extract_sql_query(x["query"]))
        if extract_sql_query(x["query"])
        else "Error: No se pudo extraer una consulta SQL válida."
    )
    | answer_prompt
    | llm
    | StrOutputParser()
)

# Ejecución final
response = chain.invoke({
    "question": "List the species of trees that are present in San Francisco",
    "input": "List the species of trees that are present in San Francisco",
    "table_info": db.get_table_info(),
    "top_k": 10
})

print("\n----------\n")
print("List the species of trees that are present in San Francisco (passing question and result to the LLM)")
print("\n----------\n")
print(response)
print("\n----------\n")