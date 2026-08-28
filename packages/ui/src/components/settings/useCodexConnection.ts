"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import {
  cancelCodexDeviceLogin,
  CodexAuthRequestError,
  CodexAuthStatus,
  getCodexAuthStatus,
  startCodexDeviceLogin,
} from "@/lib/api";

const STATUS_POLL_INTERVAL_MS = 2_000;

export function useCodexConnection() {
  const [connection, setConnection] = useState<CodexAuthStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [forbidden, setForbidden] = useState(false);
  // Only the newest request may update state. A stale poll must not restore a
  // pending login after a cancellation or manual refresh.
  const requestSequence = useRef(0);

  const refresh = useCallback(async () => {
    const requestId = ++requestSequence.current;
    try {
      const next = await getCodexAuthStatus();
      if (requestId !== requestSequence.current) return;
      setConnection(next);
      setError(null);
      setForbidden(false);
    } catch (err) {
      if (requestId !== requestSequence.current) return;
      if (err instanceof CodexAuthRequestError && err.status === 403) {
        setForbidden(true);
        setConnection(null);
        setError(null);
        return;
      }
      setError(err instanceof Error ? err.message : "Could not read Codex status.");
    } finally {
      if (requestId === requestSequence.current) setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    if (connection?.state !== "pending") return;
    const timer = window.setInterval(() => void refresh(), STATUS_POLL_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [connection?.state, refresh]);

  const connect = useCallback(async () => {
    // Reserve a user-initiated tab before awaiting the API response. Opening
    // after an await is commonly blocked as an unsolicited popup.
    const signInWindow = window.open("", "_blank");
    if (signInWindow) signInWindow.opener = null;
    ++requestSequence.current;
    setBusy(true);
    setError(null);
    try {
      const login = await startCodexDeviceLogin();
      setConnection({ state: "pending", ...login });
      setLoading(false);
      if (signInWindow && !signInWindow.closed) {
        signInWindow.location.replace(login.verification_url);
      }
    } catch (err) {
      signInWindow?.close();
      await refresh();
      if (err instanceof CodexAuthRequestError) {
        if (err.status === 403) setForbidden(true);
        // Status is authoritative if another tab owns/completed the flow.
        if (err.status === 403 || err.status === 409) return;
      }
      setError(err instanceof Error ? err.message : "Could not start ChatGPT sign-in.");
    } finally {
      setBusy(false);
    }
  }, [refresh]);

  const cancel = useCallback(async () => {
    ++requestSequence.current;
    setBusy(true);
    setError(null);
    try {
      await cancelCodexDeviceLogin();
      await refresh();
    } catch (err) {
      await refresh();
      if (err instanceof CodexAuthRequestError) {
        if (err.status === 403) setForbidden(true);
        if (err.status === 403 || err.status === 409) return;
      }
      setError(err instanceof Error ? err.message : "Could not cancel ChatGPT sign-in.");
    } finally {
      setBusy(false);
    }
  }, [refresh]);

  return { connection, loading, busy, error, forbidden, connect, cancel, refresh };
}
