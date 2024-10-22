import { PremEmbeddings } from "@langchain/community/embeddings/premai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import {ChatGroq} from "@langchain/groq";
import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

let vectorstore;
let ragChain;
let history= [];
async function initializeChain() {

  const loader = new DirectoryLoader("./pdf",{
    // '.pdf': (path) => new PDFLoader(path),
    '.txt': (path) => new TextLoader(path),
  })

  const docs = await loader.load();

  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 100000,
    chunkOverlap: 0,
  });

  const splits = await textSplitter.splitDocuments(docs);

  const embeddings = new PremEmbeddings({
    apiKey: process.env.PREM_API_KEY,
    model: 'embed-multilingual-light',
    project_id: 6255,
    maxRetries: 0
  });

  vectorstore = await MemoryVectorStore.fromDocuments(splits, embeddings);
  const retriever = vectorstore.asRetriever();

  let llm=new ChatGroq({
    apiKey:process.env.GROQ_API_KEY,
    model:'llama-3.1-70b-versatile',
  });

  const prompt = ChatPromptTemplate.fromTemplate(`
    Start a Friendly conversation with the user
    Answer the following question.
    underline the main answer.

{context}
History:{chat_history}
Human: {input}
AI: Let's think about this step-by-step:

`);

  const combineDocsChain = await createStuffDocumentsChain({
    llm,
    prompt,
  });

  ragChain = await createRetrievalChain({
    retriever,
    combineDocsChain,
  });
}


initializeChain();

app.post('/chat', async (req, res) => {
  const result=await ragChain.invoke({
    input:req.body.input,
    chat_history:history.join('\n')
  })

  history.push(
    `Human: ${req.body.input}`, 
    `AI: ${result.answer}`
  );

  history=history.slice(-40);
  
  res.json({ answer: result.answer, history: history });
});

app.get("/chat", (req, res) => res.send("Hello World!"));


app.listen(3000,()=>{
    console.log("Server is running on port 3000");
})


