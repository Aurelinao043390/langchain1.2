import os
from dotenv import load_dotenv
from rich import print as rp
from openai import OpenAI
from pymilvus import  MilvusClient
from langchain_community.document_loaders import TextLoader,UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

# ================================1 全局向量初始化=========================
MILVUS_URI="http://localhost:19530"
DB_NAME="rag_tutorial"
COLLECTION_NAME="docs"
KNOWLEDGE_FILE_TXT="曲面打印机说明书.txt"
KNOWLEDGE_FILE_DOCS="实施和定制开发服务工作说明书.docx"

EMBEDDING_MODEL_NAME="qwen3.7-text-embedding"
# EMBEDDING_MODEL_NAME="qwen3-vl-embedding"

EMBED_DIM=1024# qwen3.7-text-embedding的维度可选值：2560、2048、1536、1024、768、512、256



# ==================================2 初始化Milvus===========================

#初始化客户端
milvus_client = MilvusClient(MILVUS_URI)
#查询已有数据库，如果没有就创建
existed_databases=milvus_client.list_databases()
if DB_NAME not in existed_databases:
    milvus_client.create_database(db_name=DB_NAME)
#切换到指定数据库
milvus_client.use_database(db_name=DB_NAME)
#创建collection,如果已存在指定名称的collection，为了避免冲突，需要将已有的删除，
if milvus_client.has_collection(collection_name=COLLECTION_NAME):
    milvus_client.drop_collection(collection_name=COLLECTION_NAME)
milvus_client.create_collection(
    collection_name=COLLECTION_NAME,
    dimension=EMBED_DIM,
    metric_type="COSINE",
)
# ====================================3初始化Embedding模型=======================================

load_dotenv(override=True)
my_api_key = os.getenv("ALL_API_KEY")
my_base_url = os.getenv("EMBEDDING_BASE_URL")
# 通用初始化
openai_client=OpenAI(
    api_key=my_api_key,
    base_url=my_base_url,
)

# ===================================4 加载文档并切分=============================================

#加载text文档
text_loader = TextLoader(
    file_path=KNOWLEDGE_FILE_TXT,   encoding="utf-8"
)
text_content=text_loader.load()
#加载DOCS文档
docs_loader=UnstructuredWordDocumentLoader(
    file_path=KNOWLEDGE_FILE_DOCS, encoding="utf-8"
)
docx_content=docs_loader.load()

#切分text文档
text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    #曲面打印机说明书.txt文档切分定制化策略
    separators=[
        "\n\n",
        "\n",
        "\n★",
        "。",
        "；",
        "，",
        "、",
        "|",
        "：",
        ":",
        "-",
        "",
    ]
)
text_chunks=text_splitter.split_documents(text_content)

docx_splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    #实施和定制开发服务工作说明书.docx文档切分定制化策略
    separators = [
    "\n",
    "。",
    "；",
    "，",
    "、",
    "\t",
    "|",
    "：",
    ":",
    "-",
    ".",
    ")",
    "",
]
)
docx_chunks=docx_splitter.split_documents(docx_content)
# ======================================5 生成向量并写入Milvus=======================================================

text_input=[
    chunk.page_content for chunk in text_chunks
]

docx_input=[
    chunk.page_content for chunk in docx_chunks
]
# #向量化过程

text_length=len(text_input)
docx_length=len(docx_input)
# 1. 将输入切分为每组10条（小于API限制的20条即可）
batch_size = 10
text_all_embeddings = []
docx_all_embeddings = []
# 2. 循环调用 API，每批次处理一部分文本
for i in range(0, text_length, batch_size):
    #设置每轮只取10条文本到batch里面
    batch = text_input[i : i + batch_size]

    rp(f"[bold green]正在处理第 {i//batch_size + 1} 批次... ([bold]{len(batch)} 条[/])")
    # print(len(batch))
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL_NAME,
        dimensions=EMBED_DIM,
        input=batch  # 这里只传入 10 条文本
    )

    # 3. 将当前批次的向量结果追加到总列表中
    batch_embeddings = [item.embedding for item in response.data]
    text_all_embeddings.extend(batch_embeddings)


for i in range(0, docx_length, batch_size):
    #设置每轮只取10条文本到batch里面
    batch = docx_input[i : i + batch_size]

    rp(f"[bold green]正在处理第 {i//batch_size + 1} 批次... ([bold]{len(batch)} 条[/])")
    # print(len(batch))
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL_NAME,
        dimensions=EMBED_DIM,
        input=batch  # 这里只传入 10 条文本
    )

    # 3. 将当前批次的向量结果追加到总列表中
    batch_embeddings = [item.embedding for item in response.data]
    docx_all_embeddings.extend(batch_embeddings)

#构建对应的数据
text_data=[
    {
            "id":i,
            "chunk_id":i,
            "vector":text_all_embeddings[i],
            "text":text_chunks[i].page_content,
            "source":KNOWLEDGE_FILE_TXT
    }
    for i in range(len(text_chunks))
]
docx_data=[
     {
            "id": len(text_chunks) + i,
            "chunk_id": len(text_chunks) + i,
            "vector":docx_all_embeddings[i],
            "text":docx_chunks[i].page_content,
            "source":KNOWLEDGE_FILE_DOCS
    }
    for i in range(len(docx_chunks))
]
text_insert_result=milvus_client.upsert(
    collection_name=COLLECTION_NAME,
    data=text_data,
)

docx_insert_result=milvus_client.upsert(
    collection_name=COLLECTION_NAME,
    data=docx_data,
)
print("text_insert_result:",text_insert_result)
print("docx_insert_result:",docx_insert_result)
#flush磁盘
milvus_client.flush(collection_name=COLLECTION_NAME)
# 打印当前集合中的统计信息
# stats=milvus_client.get_collection_stats(collection_name=COLLECTION_NAME)
# print(stats)
# ======================================6 创建Agent==================================================
#1 初始化思考大模型
myModel= init_chat_model(
    model="qwen3.8-max",
    model_provider="openai",
    api_key=my_api_key,
    base_url=my_base_url,
)
#初始化Agent
myAgent=create_agent(
    model=myModel,

)
# =====================================7 检索功能=======================================================
#定义1个具体的函数，实现检索
def retrieve(user_query:str,limit:int=3):

    #将用户的问题向量化
    response=openai_client.embeddings.create(
        model="qwen3.7-text-embedding",
        dimensions=1024,
        input=user_query

    )
    #假设用户的问题只有1句话
    query_vector=response.data[0].embedding
    #从向量数据库中检索数据
    results=milvus_client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        limit=limit,
        output_fields=["id", "chunk_id","text","source"]
    )
    return results[0]
# ====================================8 回答问题功能===========================================
def generate_answer(user_query: str):
    # 检索到的数据
    hits = retrieve(user_query)
    # 格式化数据
    context_blocks = []
    print("==============检索结果：==============")
    for i, hit in enumerate(hits):
        text = hit["entity"]["text"]
        source = hit["entity"]["source"]
        chunk_id = hit["entity"]["chunk_id"]
        score = hit["distance"]  # Cosine模式下，score越高表示越相似
        print(f"[{i}] chunk_id={chunk_id}  source={source} score={score:.4f}")
        print(text)
        print()
        # 拼接成带有编号和元数据的规范上下文块
        context_blocks.append(
            # f"[片段{i} | chunk_id={chunk_id} | source={source} \n {text}]"
            f"[参考片段{i} \n {text}]"
        )
    # 将多个上下文片段用换行符连成1个大字符串
    context = "\n\n".join(context_blocks)
    # 构造1个agent需要用的Prompt

    # 【回答要求】：
    # 1. 必须只根据【已知信息】回答，不能凭空编造。
    # 2. 如果【已知信息】中完全没有包含回答【问题】所需的内容，请明确回答“根据当前已知信息，我无法回答该问题”。
    # 3. 回答语言请使用简体中文。
    # 调用Agent
    system_prompt = f""""
            你是一个打印机操作助手。
            请忽略你自身已有的知识，**严格且仅仅根据下面的 <context> 标签内的文本内容**来回答用户的 <query>。

            <context>
            {context}
            </context>

            <query>
            {user_query}
            </query>

            【回答要求】
            1. 如果 <context> 标签内包含了回答 <query> 所需的确切信息，请直接给出答案。
            2. 如果 <context> 标签内完全没有提及 <query> 相关信息，请直接回答：“根据提供的已知信息，无法回答该问题。”，不要解释，不要道歉。
    """
    response = myAgent.invoke(
        {
            "messages":
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_query)
                ]
        }
    )
    final_msg = response["messages"][-1]
    # #最终回答
    print("===============最终回答===============")
    final_msg.pretty_print()

if __name__=="__main__":
    my_query = "打印机最大负载的额定电流？"
    generate_answer(my_query)