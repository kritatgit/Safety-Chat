import React, { useMemo, useState, useCallback,useEffect } from "react";

// Token limits (must match backend). Approx 4 chars per token for UI cap.
const USER_PROMPT_MAX_TOKENS = 1000;
const SYSTEM_PROMPT_MAX_TOKENS = 500;
const CHARS_PER_TOKEN_APPROX = 4;
const MAX_RECENT_MESSAGES = 10; // Keep last 10 messages in context (5 pairs)
const SUMMARIZE_AFTER = 16; // Trigger summarization after 16 messages (8 pairs)
const SUMMARIZE_BATCH = 6; // Summarize 6 oldest messages at a time

const userPromptMaxChars = USER_PROMPT_MAX_TOKENS * CHARS_PER_TOKEN_APPROX;
const systemPromptMaxChars = SYSTEM_PROMPT_MAX_TOKENS * CHARS_PER_TOKEN_APPROX;

function approxTokens(text: string): number {
  return Math.ceil(text.length / CHARS_PER_TOKEN_APPROX);
}

type ChatResponse = {
  prompt: string;
  label: "safe" | "unsafe";
  safe_probability: number;
  unsafe_probability: number;
  model: string;
  assistant: string;
  summary?: string; // New field for summary response
};

type Msg = { role: "user" | "assistant"; content: string };

export function App() {
  const [systemPrompt, setSystemPrompt] = useState<string>(
    "You are a helpful assistant."
  );
  const [includeSafety, setIncludeSafety] = useState<boolean>(true);
  const [input, setInput] = useState<string>("");

  // SEPARATE STATE: Display messages (never trimmed, always grows)
  const [displayMessages, setDisplayMessages] = useState<Msg[]>([]);

  // SEPARATE STATE: Context messages (trimmed for API calls)
  const [contextMessages, setContextMessages] = useState<Msg[]>([]);
// Auto-trigger summarization when context grows too large

  const [summary, setSummary] = useState<string | null>(null);
  const [lastMeta, setLastMeta] = useState<ChatResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>("");

  const canSend = useMemo(
    () => input.trim().length > 0 && !loading,
    [input, loading]
  );
  useEffect(() => {
    if (contextMessages.length > SUMMARIZE_AFTER && !loading) {
      summarizeOldContext();
    }
  }, [contextMessages.length, loading]); // Only trigger when length changes
  // Get messages to send (for context management)
  const getRecentMessages = useCallback(() => {
    // Keep only last MAX_RECENT_MESSAGES from context
    if (contextMessages.length > MAX_RECENT_MESSAGES) {
      return contextMessages.slice(-MAX_RECENT_MESSAGES);
    }
    return contextMessages;
  }, [contextMessages]);

  // Summarize old context
  const summarizeOldContext = useCallback(async () => {
    if (contextMessages.length <= SUMMARIZE_AFTER) return;

    try {
      // Take first SUMMARIZE_BATCH messages to summarize
      const toSummarize = contextMessages.slice(0, SUMMARIZE_BATCH);

      // Build system prompt for summarization
      let summarySystemPrompt: string;
      if (summary) {
        summarySystemPrompt = `You are summarizing a conversation.

Here is the existing summary:
${summary}

Now add information from these new messages to create an updated summary.
Keep it concise (under 150 words) but preserve all key facts, decisions, and context.`;
      } else {
        summarySystemPrompt =
          "Summarize the following conversation concisely, keeping key facts and context. Keep it under 100 words.";
      }

      // Format messages for summarization
      const formattedMessages = toSummarize
        .map((m) => `${m.role.toUpperCase()}: ${m.content}`)
        .join("\n\n");

      // Call backend with summarize=true
      const r = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: formattedMessages,
          system_prompt: summarySystemPrompt,
          include_safety_in_system_prompt: false,
          summarize: true, // Flag to indicate this is a summary request
        }),
      });

      if (!r.ok) {
        console.error("Summarization failed");
        return;
      }

      const data = (await r.json()) as ChatResponse;

      // Update summary
      setSummary(data.assistant);

      // Remove summarized messages from CONTEXT only (NOT from display)
      setContextMessages((prev) => prev.slice(SUMMARIZE_BATCH));
    } catch (e) {
      console.error("Error during summarization:", e);
    }
  }, [contextMessages, summary]);

  async function send() {
    const prompt = input.trim();
    if (!prompt || loading) return;

    setError("");
    setLoading(true);
    setInput("");

    // Add user message to BOTH display and context
    const newUserMessage: Msg = { role: "user", content: prompt };
    setDisplayMessages((m) => [...m, newUserMessage]);
    setContextMessages((m) => [...m, newUserMessage]);

    try {
      const r = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          system_prompt: systemPrompt,
          include_safety_in_system_prompt: includeSafety,
          context: getRecentMessages(), // Send recent messages as context
          summary: summary, // Send existing summary
        }),
      });

      if (!r.ok) {
        const text = await r.text();
        let message: string;
        try {
          const err = JSON.parse(text) as { detail?: unknown };
          if (Array.isArray(err.detail) && err.detail.length > 0) {
            const first = err.detail[0] as { msg?: string; type?: string };
            message =
              first.msg ??
              (typeof err.detail[0] === "string" ? err.detail[0] : text);
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

      // Add assistant message to BOTH display and context
      const newAssistantMessage: Msg = {
        role: "assistant",
        content: data.assistant,
      };
      setDisplayMessages((m) => [...m, newAssistantMessage]);
      setContextMessages((m) => [...m, newAssistantMessage]);

      // Check if we need to summarize (using contextMessages length)
      // Use setTimeout to avoid blocking the UI
      // setTimeout(() => {
      //   // Check against the updated contextMessages length
      //   setContextMessages((currentContext) => {
      //     if (currentContext.length > SUMMARIZE_AFTER) {
      //       // Trigger summarization
      //       summarizeOldContext();
      //     }
      //     return currentContext;
      //   });
      // }, 100);
    } catch (e: any) {
      setError(e?.message ?? String(e));
      // Remove the user message that failed from BOTH
      setDisplayMessages((m) => m.slice(0, -1));
      setContextMessages((m) => m.slice(0, -1));
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
          User prompt → <code>/predict</code> → GPT response (with optional
          custom system prompt)
        </div>
      </header>

      <div className="grid">
        <section className="panel">
          <div className="panelTitle">System prompt</div>
          <textarea
            className="textarea"
            value={systemPrompt}
            onChange={(e) => {
              const v = e.target.value;
              if (v.length <= systemPromptMaxChars) setSystemPrompt(v);
            }}
            rows={8}
            placeholder="Enter a custom system prompt..."
            maxLength={systemPromptMaxChars}
          />
          <div className="hint" style={{ marginTop: 6 }}>
            Approx. tokens: {approxTokens(systemPrompt)}/
            {SYSTEM_PROMPT_MAX_TOKENS}
          </div>
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
            {displayMessages.length === 0 ? (
              <div className="empty">
                Type a prompt below. Press <code>Ctrl</code>+<code>Enter</code>{" "}
                to send.
              </div>
            ) : (
              displayMessages.map((m, idx) => (
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
              onChange={(e) => {
                const v = e.target.value;
                if (v.length <= userPromptMaxChars) setInput(v);
              }}
              onKeyDown={onKeyDown}
              rows={3}
              placeholder="Enter a prompt..."
              disabled={loading}
              maxLength={userPromptMaxChars}
            />
            <button className="button" disabled={!canSend} onClick={send}>
              {loading ? "Sending..." : "Send"}
            </button>
          </div>
          <div className="hint">
            Approx. tokens: {approxTokens(input)}/{USER_PROMPT_MAX_TOKENS}.{" "}
            Tip: <code>Ctrl</code>+<code>Enter</code> to send.
          </div>
        </section>
      </div>
    </div>
  );
}