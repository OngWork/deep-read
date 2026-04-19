import React, { useState } from 'react';
import { Upload, X, Send, FileText, Loader2 } from 'lucide-react';
import axios from 'axios';

function App() {
  const [files, setFiles] = useState([]); // keep track of uploaded files
  const [query, setQuery] = useState(""); // keep track of user input
  const [chatHistory, setChatHistory] = useState([
    { role: 'ai', text: 'Hello I am DeepRead AI, please upload a PDF file and start asking questions!' }
  ]);
  const [isLoading, setIsLoading] = useState(false); // state to track loading status
  const API_BASE_URL = "http://3.25.245.209:8000"; // public IP of EC2 instance

  // --- 1. upload file function ---
  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setIsLoading(true); // start loading
    const formData = new FormData();
    formData.append("file", file);

    try {
      // upload to s3
      const res = await axios.post(`${API_BASE_URL}/upload`, formData);

      if (res.data.status === "Success") {
        const filename = res.data.filename;
        // send file to pinecone for processing
        await axios.post(`${API_BASE_URL}/process-s3-file?filename=${filename}`);
        
        setFiles(prev => [...prev, file.name]);
        alert("Analyzed file successfully");
      }
    } catch (err) {
      console.error("Upload or Process failed", err);
      alert("An error occurred while processing the file!");
    } finally {
      setIsLoading(false); // cancel loading
    }
  };

  // --- 2. chat AI function ---
  const handleChat = async () => {
    if (!query.trim() || isLoading) return;

    // Add user message to chat history immediately for better UX
    const userMessage = { role: 'user', text: query };
    setChatHistory(prev => [...prev, userMessage]);
    setQuery("");
    setIsLoading(true);

    const formData = new FormData();
    formData.append("query", query);
    formData.append("history", JSON.stringify(chatHistory)); // add chat history to provide context for the AI

    const res = await axios.post(`${API_BASE_URL}/upload`, formData);

    try {
      const res = await axios.post(`${API_BASE_URL}/upload`, formData);
      
      const aiResponse = {
        role: 'ai',
        text: res.data.answer,
        sources: res.data.sources // keep reference to sources for later display
      };
      
      setChatHistory(prev => [...prev, aiResponse]);
    }finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-screen w-full bg-gray-50 text-gray-900 overflow-hidden font-sans">
      
      {/* Sidebar */}
      <aside className="w-72 bg-white border-r border-gray-200 flex flex-col flex-shrink-0">
        <div className="p-6 border-b">
          <h2 className="text-xl font-bold mb-4 flex items-center gap-2">DeepRead AI 📚</h2>
          <label className="flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 text-white py-3 px-4 rounded-xl cursor-pointer transition-all shadow-md active:scale-95">
            <Upload size={20} />
            <span className="font-medium">Upload PDF</span>
            <input type="file" className="hidden" onChange={handleFileUpload} accept=".pdf" />
          </label>
        </div>
        
        <div className="flex-1 overflow-y-auto p-4">
          <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-4 px-2">Your Documents</p>
          <div className="space-y-1">
            {files.map((file, index) => (
              <div key={index} className="flex items-center justify-between p-3 hover:bg-gray-100 rounded-xl group transition">
                <div className="flex items-center gap-3 truncate">
                  <FileText size={18} className="text-blue-500" />
                  <span className="text-sm font-medium truncate">{file}</span>
                </div>
                <button className="text-gray-400 hover:text-red-500 opacity-0 group-hover:opacity-100 transition">
                  <X size={16} />
                </button>
              </div>
            ))}
          </div>
        </div>
      </aside>

      {/* Chat Area */}
      <main className="flex-1 flex flex-col bg-slate-50 relative">
        {/* --- Chat History --- */}
        <section className="flex-1 overflow-y-auto p-8 space-y-6 pb-32">
          {chatHistory.map((msg, i) => (
            <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              
              {/* Chat Bubble */}
              <div className={`max-w-[80%] px-5 py-3 rounded-2xl shadow-sm border ${
                msg.role === 'user' 
                ? 'bg-blue-600 border-blue-600 text-white rounded-tr-none' 
                : 'bg-white border-gray-200 text-gray-800 rounded-tl-none'
              }`}>
                
                {/* --- Content Section --- */}
                <p className="text-sm md:text-base whitespace-pre-wrap">
                  {typeof msg.text === 'object' ? (msg.text.content || JSON.stringify(msg.text)) : msg.text}
                </p>
                
                {/* --- New Section: Source Citations (append after the message) --- */}
                {msg.sources && msg.sources?.length > 0 && (
                  <div className="mt-3 pt-2 border-t border-gray-100 flex flex-wrap gap-2">
                    <span className="text-[10px] font-bold text-gray-400 uppercase">Sources:</span>
                    {msg.sources.map((s, idx) => (
                      <span key={idx} className="bg-blue-50 text-blue-600 text-[10px] px-2 py-0.5 rounded-full font-medium border border-blue-100">
                        📄 {s?.file?.replace('temp_', '')} (P.{s?.page})
                      </span>
                    ))}
                  </div>
                )}

              </div>
            </div>
          ))}
        </section>

        {/* Input Box */}
        <div className="absolute bottom-0 left-0 w-full p-8 bg-gradient-to-t from-slate-50 via-slate-50 to-transparent">
          <form 
            onSubmit={(e) => { e.preventDefault(); handleChat(); }}
            className="max-w-4xl mx-auto flex gap-3 items-center bg-white p-2 rounded-2xl shadow-xl border border-gray-200"
          >
            <input 
              type="text" 
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask anything about the uploaded file?" 
              className="flex-1 pl-4 py-3 bg-transparent outline-none text-gray-700"
            />
            <button 
              type="submit"
              disabled={isLoading}
              className={`p-3 rounded-xl transition shadow-md active:scale-90 ${
                isLoading ? 'bg-gray-300' : 'bg-blue-600 text-white hover:bg-blue-700'
              }`}
            >
              <Send size={20} />
            </button>
          </form>
        </div>
      </main>
    </div>
  );
}

export default App;