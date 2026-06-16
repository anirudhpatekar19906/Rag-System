import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import {
  Send,
  FileText,
  Youtube,
  Github,
  Loader2,
  Database,
  User,
  Bot,
  PlusCircle,
  ExternalLink
} from 'lucide-react';

const API_BASE = 'http://127.0.0.1:8000';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState({ index_size: 0 });
  const [ingesting, setIngesting] = useState(false);

  // Ingestion states
  const [pdfFile, setPdfFile] = useState(null);
  const [youtubeUrl, setYoutubeUrl] = useState('');
  const [githubUrl, setGithubUrl] = useState('');

  const scrollRef = useRef(null);

  useEffect(() => {
    fetchStatus();
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const fetchStatus = async () => {
    try {
      const res = await axios.get(`${API_BASE}/status`);
      setStatus(res.data);
    } catch (e) {
      console.error("Failed to fetch status", e);
    }
  };

  const handleIngestPdf = async () => {
    if (!pdfFile) return;
    setIngesting(true);
    const formData = new FormData();
    formData.append('file', pdfFile);
    try {
      await axios.post(`${API_BASE}/ingest-pdf`, formData);
      alert('PDF ingested successfully!');
      fetchStatus();
    } catch (e) {
      const errorMsg = e.response?.data?.detail || e.message || 'Unknown error';
      alert('Error ingesting PDF: ' + errorMsg);
    } finally {
      setIngesting(false);
      setPdfFile(null);
    }
  };

  const handleIngestLink = async () => {
    if (!youtubeUrl && !githubUrl) return;
    setIngesting(true);
    try {
      await axios.post(`${API_BASE}/ingest-link`, {
        youtube_url: youtubeUrl,
        github_url: githubUrl
      });
      alert('Link ingested successfully!');
      fetchStatus();
      setYoutubeUrl('');
      setGithubUrl('');
    } catch (e) {
      const errorMsg = e.response?.data?.detail || e.message || 'Unknown error';
      alert('Error ingesting link: ' + errorMsg);
    } finally {
      setIngesting(false);
    }
  };

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMsg = input;
    setInput('');
    setMessages(prev => [...prev, { role: 'user', text: userMsg }]);
    setLoading(true);

    try {
      const res = await axios.post(`${API_BASE}/query`, { question: userMsg });
      const { answer, sources } = res.data;
      setMessages(prev => [...prev, { role: 'bot', text: answer, sources }]);
    } catch (e) {
      setMessages(prev => [...prev, { role: 'bot', text: 'Error: Unable to get a response. The server might be overloaded (503).' }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-screen bg-darkest text-slate-200 font-sans overflow-hidden">
      {/* SIDEBAR */}
      <aside className="w-80 bg-dark border-r border-slate-800 flex flex-col p-6 transition-all">
        <div className="flex items-center gap-2 mb-8">
          <Database className="text-accent" size={24} />
          <h1 className="text-xl font-bold tracking-tight">Multi-RAG <span className="text-accent">AI</span></h1>
        </div>

        <div className="space-y-6">
          {/* Status Card */}
          <div className="p-4 rounded-xl bg-slate-800/50 border border-slate-700">
            <div className="text-xs text-slate-400 uppercase font-semibold mb-1">Index Status</div>
            <div className="text-lg font-medium">{status.index_size} <span className="text-sm text-slate-500">chunks indexed</span></div>
          </div>

          {/* PDF Upload */}
          <div className="space-y-3">
            <label className="text-sm font-medium text-slate-400 flex items-center gap-2">
              <FileText size={16} /> Upload Document
            </label>
            <div className="flex flex-col gap-2">
              <input
                type="file"
                onChange={(e) => setPdfFile(e.target.files[0])}
                className="block w-full text-sm text-slate-400
                  file:mr-4 file:py-2 file:px-4
                  file:rounded-full file:border-0
                  file:text-sm file:font-semibold
                  file:bg-accent file:text-darkest
                  hover:file:bg-sky-300 cursor-pointer"
              />
              <button
                onClick={handleIngestPdf}
                disabled={!pdfFile || ingesting}
                className="w-full py-2 bg-slate-700 hover:bg-slate-600 disabled:opacity-50 rounded-lg text-sm font-medium transition-colors flex items-center justify-center gap-2"
              >
                {ingesting ? <Loader2 className="animate-spin" size={16} /> : <PlusCircle size={16} />}
                Ingest PDF
              </button>
            </div>
          </div>

          {/* Links Upload */}
          <div className="space-y-3">
            <label className="text-sm font-medium text-slate-400 flex items-center gap-2">
              <Youtube size={16} /> Web Sources
            </label>
            <div className="flex flex-col gap-2">
              <input
                type="text"
                placeholder="YouTube URL..."
                value={youtubeUrl}
                onChange={(e) => setYoutubeUrl(e.target.value)}
                className="w-full px-3 py-2 rounded-lg bg-slate-800 border border-slate-700 text-sm focus:outline-none focus:ring-2 focus:ring-accent"
              />
              <input
                type="text"
                placeholder="GitHub Repo..."
                value={githubUrl}
                onChange={(e) => setGithubUrl(e.target.value)}
                className="w-full px-3 py-2 rounded-lg bg-slate-800 border border-slate-700 text-sm focus:outline-none focus:ring-2 focus:ring-accent"
              />
              <button
                onClick={handleIngestLink}
                disabled={(!youtubeUrl && !githubUrl) || ingesting}
                className="w-full py-2 bg-slate-700 hover:bg-slate-600 disabled:opacity-50 rounded-lg text-sm font-medium transition-colors flex items-center justify-center gap-2"
              >
                {ingesting ? <Loader2 className="animate-spin" size={16} /> : <PlusCircle size={16} />}
                Ingest Links
              </button>
            </div>
          </div>
        </div>

        <div className="mt-auto pt-6 text-xs text-slate-500 text-center">
          Powered by Gemini 1.5 Flash
        </div>
      </aside>

      {/* MAIN CHAT AREA */}
      <main className="flex-1 flex flex-col relative h-full">
        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {messages.length === 0 && (
            <div className="h-full flex flex-col items-center justify-center text-center space-y-4 opacity-50">
              <div className="p-4 rounded-full bg-slate-800">
                <Bot size={48} className="text-slate-600" />
              </div>
              <div>
                <h2 className="text-xl font-semibold">How can I help you today?</h2>
                <p className="text-sm">Upload a document to start asking questions.</p>
              </div>
            </div>
          )}

          {messages.map((msg, i) => (
            <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'} animate-in fade-in slide-in-from-bottom-2`}>
              <div className={`max-w-2xl flex gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
                <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${msg.role === 'user' ? 'bg-accent text-darkest' : 'bg-slate-700 text-slate-300'}`}>
                  {msg.role === 'user' ? <User size={16} /> : <Bot size={16} />}
                </div>
                <div className="space-y-2">
                  <div className={`p-4 rounded-2xl ${msg.role === 'user' ? 'bg-accent text-darkest rounded-tr-none' : 'bg-slate-800 border border-slate-700 rounded-tl-none'}`}>
                    <p className="text-sm leading-relaxed whitespace-pre-wrap">{msg.text}</p>
                  </div>
                  {msg.sources && (
                    <div className="flex flex-wrap gap-2 pl-1">
                      {msg.sources.map((s, si) => (
                        <span key={si} className="text-[10px] px-2 py-0.5 rounded-full bg-slate-800 border border-slate-700 text-slate-400 hover:text-accent cursor-default flex items-center gap-1 transition-colors">
                          <ExternalLink size={10} /> {s.source}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
          {loading && (
            <div className="flex justify-start">
              <div className="max-w-2xl flex gap-3">
                <div className="w-8 h-8 rounded-full bg-slate-700 flex items-center justify-center shrink-0 text-slate-300">
                  <Bot size={16} />
                </div>
                <div className="p-4 rounded-2xl bg-slate-800 border border-slate-700 rounded-tl-none">
                  <Loader2 className="animate-spin text-accent" size={20} />
                </div>
              </div>
            </div>
          )}
          <div ref={scrollRef} />
        </div>

        {/* Input Area */}
        <div className="p-6 bg-gradient-to-t from-darkest to-transparent">
          <form onSubmit={handleSend} className="max-w-4xl mx-auto relative">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask something about your documents..."
              className="w-full px-6 py-4 pr-14 rounded-2xl bg-slate-800 border border-slate-700 focus:outline-none focus:ring-2 focus:ring-accent text-slate-200 placeholder-slate-500 shadow-2xl"
            />
            <button
              type="submit"
              disabled={!input.trim() || loading}
              className="absolute right-3 top-1/2 -translate-y-1/2 p-2 bg-accent text-darkest rounded-xl hover:bg-sky-300 disabled:opacity-50 transition-all"
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
