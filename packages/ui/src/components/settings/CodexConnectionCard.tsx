"use client";

import Icon from "@/components/Icon";
import { CodexAuthStatus } from "@/lib/api";

import { useCodexConnection } from "./useCodexConnection";

function PendingLogin({
  connection,
  busy,
  onCancel,
}: {
  connection: CodexAuthStatus;
  busy: boolean;
  onCancel: () => Promise<void>;
}) {
  return (
    <div className="mt-4 rounded-lg border border-indigo-500/30 bg-indigo-500/10 p-4">
      <p className="text-xs font-medium text-indigo-200">Finish signing in</p>
      <p className="mt-1 text-xs text-fg-muted">
        Open the ChatGPT sign-in page and enter this one-time code:
      </p>
      <div className="mt-3 flex flex-wrap items-center gap-3">
        <code className="select-all rounded-md border border-line-strong bg-surface px-3 py-2 text-base tracking-[0.2em] text-fg">
          {connection.user_code}
        </code>
        {connection.verification_url && (
          <a
            href={connection.verification_url}
            target="_blank"
            rel="noopener noreferrer"
            className="rounded-lg bg-indigo-500 px-3.5 py-2 text-xs font-medium text-white hover:bg-indigo-400"
          >
            Open sign-in page
          </a>
        )}
        <button
          type="button"
          onClick={() => void onCancel()}
          disabled={busy}
          className="rounded-lg border border-line px-3.5 py-2 text-xs text-fg-muted hover:border-line-strong hover:text-fg disabled:opacity-50"
        >
          {busy ? "Cancelling…" : "Cancel"}
        </button>
      </div>
      <p className="mt-3 text-[11px] text-fg-muted">
        This page checks for completion automatically. The code expires if sign-in is not completed.
      </p>
    </div>
  );
}

function ConnectedAccount({ connection }: { connection: CodexAuthStatus }) {
  const chatgpt = connection.auth_mode === "chatgpt";
  const details = [connection.email, connection.plan_type, connection.auth_mode]
    .filter(Boolean)
    .join(" · ");
  return (
    <div className="mt-4 rounded-lg border border-line bg-surface px-4 py-3 text-xs">
      <p className="font-medium text-fg">
        {chatgpt ? "ChatGPT connected" : "Codex is already authenticated"}
      </p>
      {details && <p className="mt-1 text-fg-muted">{details}</p>}
    </div>
  );
}

function ConnectionAction({
  connection,
  busy,
  onConnect,
  onRefresh,
}: {
  connection: CodexAuthStatus;
  busy: boolean;
  onConnect: () => Promise<void>;
  onRefresh: () => Promise<void>;
}) {
  if (connection.state === "disconnected") {
    return (
      <div className="mt-4">
        <button
          type="button"
          onClick={() => void onConnect()}
          disabled={busy}
          className="rounded-lg bg-indigo-500 px-3.5 py-2 text-xs font-medium text-white hover:bg-indigo-400 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {busy ? "Starting…" : "Connect ChatGPT"}
        </button>
        <p className="mt-2 text-[11px] text-fg-muted">
          Principal-only. With `CODEX_ENABLED=true`, your subscription&apos;s models appear in Agent Council.
        </p>
      </div>
    );
  }

  if (connection.state === "error" || connection.state === "unavailable") {
    const retrySignIn = connection.state === "error";
    return (
      <div className="mt-4">
        <p className="text-xs text-rose-300">
          {connection.error ?? "Codex connection is unavailable."}
        </p>
        <button
          type="button"
          onClick={() => void (retrySignIn ? onConnect() : onRefresh())}
          disabled={busy}
          className="mt-2 rounded-lg border border-line px-3 py-1.5 text-xs text-fg-muted hover:text-fg disabled:opacity-50"
        >
          {busy ? "Starting…" : retrySignIn ? "Try sign-in again" : "Retry"}
        </button>
      </div>
    );
  }

  return null;
}

export default function CodexConnectionCard() {
  const state = useCodexConnection();
  const connectedWithChatGPT =
    state.connection?.state === "connected" && state.connection.auth_mode === "chatgpt";

  if (state.forbidden) return null;

  return (
    <section className="mt-6 rounded-xl border border-line bg-surface-elevated p-5">
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2.5">
            <span className="text-fg-muted"><Icon name="bolt" size="w-5 h-5" /></span>
            <h2 className="text-sm font-medium text-fg">ChatGPT subscription</h2>
          </div>
          <p className="mt-2 max-w-xl text-xs leading-relaxed text-fg-muted">
            Connect the principal&apos;s ChatGPT account through OpenAI&apos;s official
            Codex device sign-in. Your password is never sent to Open Executive; OAuth tokens stay inside Codex App Server.
          </p>
        </div>
        {connectedWithChatGPT && (
          <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2.5 py-1 text-xs text-emerald-300">
            <Icon name="check-circle" size="w-3.5 h-3.5" /> Connected
          </span>
        )}
      </div>

      {state.loading && <p className="mt-4 text-xs text-fg-muted">Checking connection…</p>}
      {!state.loading && state.connection?.state === "pending" && (
        <PendingLogin connection={state.connection} busy={state.busy} onCancel={state.cancel} />
      )}
      {!state.loading && state.connection?.state === "connected" && (
        <ConnectedAccount connection={state.connection} />
      )}
      {!state.loading && state.connection && (
        <ConnectionAction
          connection={state.connection}
          busy={state.busy}
          onConnect={state.connect}
          onRefresh={state.refresh}
        />
      )}
      {!state.loading && !state.connection && (
        <button
          type="button"
          onClick={() => void state.refresh()}
          className="mt-3 rounded-lg border border-line px-3 py-1.5 text-xs text-fg-muted hover:text-fg"
        >
          Retry status
        </button>
      )}
      {state.error && <p role="alert" className="mt-3 text-xs text-rose-300">{state.error}</p>}
    </section>
  );
}
