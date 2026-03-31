from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "أنت مساعد ذكي ومفيد. أجب على أسئلة المستخدم بدقة ووضوح."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])
