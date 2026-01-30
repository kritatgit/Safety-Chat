import React, { useMemo, useState } from "react";

type ChatResponse = {
  prompt: string;
  label: "safe" | "unsafe";
  safe_probability: number;
  unsafe_probability: number;
  model: string;
  assistant: string;
};

type Msg = { role: "user" | "assistant"; content: string };

export function App() {
  const [systemPrompt, setSystemPrompt] = useState<string>(
    "You are a helpful assistant."
  );
  const [includeSafety, setIncludeSafety] = useState<boolean>(true);
  const [input, setInput] = useState<string>("");
  const [messages, setMessages] = useState<Msg[]>([]);
  const [lastMeta, setLastMeta] = useState<ChatResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>("");

  const canSend = useMemo(
    () => input.trim().length > 0 && !loading,
    [input, loading]
  );

  async function send() {
    const prompt = input.trim();
    if (!prompt || loading) return;

    setError("");
    setLoading(true);
    setInput("");
    setMessages((m) => [...m, { role: "user", content: prompt }]);

    try {
      const r = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          system_prompt: systemPrompt,
          include_safety_in_system_prompt: includeSafety
        })
      });

      if (!r.ok) {
        const text = await r.text();
        let message: string;
        try {
          const err = JSON.parse(text) as { detail?: unknown };
          if (Array.isArray(err.detail) && err.detail.length > 0) {
            const first = err.detail[0] as { msg?: string; type?: string };
            message = first.msg ?? (typeof err.detail[0] === "string" ? err.detail[0] : text);
          } else if (typeof err.detail === "string") {
            message = err.detail;
          } else {
            message = text || `Request failed (${r.status})`;
          }
        } catch {
          message = text || `Request failed (${r.status})`;
        }
        throw new Error(message);
      }

      const data = (await r.json()) as ChatResponse;
      setLastMeta(data);
      setMessages((m) => [...m, { role: "assistant", content: data.assistant }]);
    } catch (e: any) {
      setError(e?.message ?? String(e));
    } finally {
      setLoading(false);
    }
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") send();
  }

  return (
    <div className="page">
      <header className="header">
        <div className="title">Prompt Safety Chat</div>
        <div className="subtitle">
          User prompt → <code>/predict</code> → GPT response (with optional custom
          system prompt)
        </div>
      </header>

      <div className="grid">
        <section className="panel">
          <div className="panelTitle">System prompt</div>
          <textarea
            className="textarea"
            value={systemPrompt}
            onChange={(e) => setSystemPrompt(e.target.value)}
            rows={8}
            placeholder="Enter a custom system prompt..."
          />
          <label className="checkboxRow">
            <input
              type="checkbox"
              checked={includeSafety}
              onChange={(e) => setIncludeSafety(e.target.checked)}
            />
            Include safety scores in system prompt
          </label>

          <div className="panelTitle" style={{ marginTop: 16 }}>
            Last safety result
          </div>
          <div className="metaBox">
            {lastMeta ? (
              <>
                <div>
                  <span className="badge">{lastMeta.label}</span>
                  <span className="muted"> model:</span> {lastMeta.model}
                </div>
                <div className="metaRow">
                  <div>
                    <span className="muted">safe:</span>{" "}
                    {lastMeta.safe_probability.toFixed(4)}
                  </div>
                  <div>
                    <span className="muted">unsafe:</span>{" "}
                    {lastMeta.unsafe_probability.toFixed(4)}
                  </div>
                </div>
              </>
            ) : (
              <div className="muted">Send a message to see scores.</div>
            )}
          </div>
        </section>

        <section className="chat">
          <div className="messages">
            {messages.length === 0 ? (
              <div className="empty">
                Type a prompt below. Press <code>Ctrl</code>+<code>Enter</code>{" "}
                to send.
              </div>
            ) : (
              messages.map((m, idx) => (
                <div
                  key={idx}
                  className={`msg ${m.role === "user" ? "user" : "assistant"}`}
                >
                  <div className="msgRole">{m.role}</div>
                  <div className="msgContent">{m.content}</div>
                </div>
              ))
            )}
          </div>

          {error ? <div className="error">{error}</div> : null}

          <div className="composer">
            <textarea
              className="textarea"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              rows={3}
              placeholder="Enter a prompt..."
              disabled={loading}
            />
            <button className="button" disabled={!canSend} onClick={send}>
              {loading ? "Sending..." : "Send"}
            </button>
          </div>
          <div className="hint">
            Tip: <code>Ctrl</code>+<code>Enter</code> to send.
          </div>
        </section>
      </div>
    </div>
  );
}

