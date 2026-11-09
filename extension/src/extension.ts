// Mine — next-edit predictions from your own self-hosted model.
// Every suggestion you accept or reject is sent back to your server, which
// periodically fine-tunes the model on it.
import * as vscode from "vscode";
import { diffLines, structuredPatch } from "diff";

// ------------------------------------------------------------------ types
interface Suggestion {
  id: string;
  model: string;
  region_start: number;
  region_end: number;
  original: string;
  suggestion: string;
}

interface Pending {
  s: Suggestion;
  doc: vscode.TextDocument;
  kind: "inline" | "edit";
  pos: vscode.Position;      // inline: where ghost text starts
  remainder: string;         // inline: insertion text not typed yet
  lineCount: number;         // document line count when shown
}

// ------------------------------------------------------------------ state
let ctx: vscode.ExtensionContext;
let token: string | undefined;
let pending: Pending | undefined;
let inflight: AbortController | undefined;
let debounce: NodeJS.Timeout | undefined;
let applying = false;
let status: vscode.StatusBarItem;
let activeModel = "";
const resolved = new Set<string>();

const snapshots = new Map<string, string>();     // uri -> text before the current burst of edits
const burstTimers = new Map<string, NodeJS.Timeout>();
const history: { filepath: string; diff: string }[] = [];   // newest first
const recent: { uri: string; line: number }[] = [];        // recently viewed files

const removedDeco = vscode.window.createTextEditorDecorationType({
  backgroundColor: new vscode.ThemeColor("diffEditor.removedLineBackground"),
  isWholeLine: true,
});
const addedDeco = vscode.window.createTextEditorDecorationType({
  after: { color: new vscode.ThemeColor("editorGhostText.foreground"), margin: "0 0 0 1.5em", fontStyle: "italic" },
});
const regionDeco = vscode.window.createTextEditorDecorationType({
  overviewRulerColor: new vscode.ThemeColor("editorInfo.foreground"),
  overviewRulerLane: vscode.OverviewRulerLane.Right,
});

// ------------------------------------------------------------------ helpers
const cfg = () => vscode.workspace.getConfiguration("mine");
const serverUrl = () => (cfg().get<string>("serverUrl") || "").replace(/\/+$/, "");
const rel = (d: vscode.TextDocument) => vscode.workspace.asRelativePath(d.uri, false);
const isCode = (d: vscode.TextDocument) =>
  (d.uri.scheme === "file" || d.uri.scheme === "untitled" || d.uri.scheme === "vscode-remote") &&
  d.getText().length < 1_500_000;

function lines(doc: vscode.TextDocument, start: number, end: number): string[] {
  const out: string[] = [];
  for (let i = Math.max(0, start); i < Math.min(doc.lineCount, end); i++) out.push(doc.lineAt(i).text);
  return out;
}

function regionText(doc: vscode.TextDocument, start: number, end: number): string {
  return lines(doc, start, end).join("\n");
}

const norm = (s: string) => s.replace(/\s+$/gm, "").replace(/^\n+|\n+$/g, "");

function makeDiff(before: string, after: string): string {
  const p = structuredPatch("a", "b", before, after, "", "", { context: 3 });
  return p.hunks
    .slice(0, 3)
    .map((h) => `@@ -${h.oldStart},${h.oldLines} +${h.newStart},${h.newLines} @@\n` + h.lines.join("\n"))
    .join("\n");
}

async function api(path: string, body?: unknown, signal?: AbortSignal): Promise<any> {
  const url = serverUrl();
  if (!url || !token) throw new Error("not configured");
  const r = await fetch(url + path, {
    method: body === undefined ? "GET" : "POST",
    headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal,
  });
  if (r.status === 401) throw new Error("bad token");
  if (!r.ok) throw new Error(`server ${r.status}`);
  return r.json();
}

function setStatus(text: string, tooltip?: string) {
  status.text = text;
  status.tooltip = tooltip;
}

function refreshStatus() {
  if (!serverUrl()) return setStatus("$(sparkle) Mine: set server", "Run 'Mine: Set Server URL'");
  if (!token) return setStatus("$(key) Mine: set token", "Run 'Mine: Set API Token'");
  if (!cfg().get<boolean>("enabled")) return setStatus("$(circle-slash) Mine off");
  setStatus(`$(sparkle) Mine${activeModel ? " · " + activeModel : ""}`, "Click to open your dashboard");
}

// ------------------------------------------------------------------ edit history
function trackChange(e: vscode.TextDocumentChangeEvent) {
  const key = e.document.uri.toString();
  if (!snapshots.has(key + "#burst")) {
    snapshots.set(key + "#burst", snapshots.get(key) ?? e.document.getText());
  }
  clearTimeout(burstTimers.get(key));
  burstTimers.set(key, setTimeout(() => flushBurst(e.document), 1500));
}

function flushBurst(doc: vscode.TextDocument) {
  const key = doc.uri.toString();
  const before = snapshots.get(key + "#burst");
  snapshots.delete(key + "#burst");
  const after = doc.getText();
  snapshots.set(key, after);
  if (before === undefined || before === after) return;
  const d = makeDiff(before, after);
  if (!d) return;
  history.unshift({ filepath: rel(doc), diff: d });
  history.splice(5);
}

function currentEdits(doc: vscode.TextDocument) {
  const key = doc.uri.toString();
  const before = snapshots.get(key + "#burst");
  const edits = [...history];
  if (before !== undefined) {
    const d = makeDiff(before, doc.getText());
    if (d) edits.unshift({ filepath: rel(doc), diff: d });
  }
  return edits.slice(0, 5);
}

function contextSnippets(doc: vscode.TextDocument) {
  const out: { filepath: string; content: string }[] = [];
  for (const r of recent) {
    if (r.uri === doc.uri.toString()) continue;
    const d = vscode.workspace.textDocuments.find((x) => x.uri.toString() === r.uri);
    if (!d) continue;
    out.push({ filepath: rel(d), content: lines(d, r.line - 20, r.line + 20).join("\n") });
    if (out.length >= 3) break;
  }
  return out;
}

function noteVisit(ed: vscode.TextEditor | undefined) {
  if (!ed || !isCode(ed.document)) return;
  const uri = ed.document.uri.toString();
  const i = recent.findIndex((r) => r.uri === uri);
  if (i >= 0) recent.splice(i, 1);
  recent.unshift({ uri, line: ed.selection.active.line });
  recent.splice(6);
}

// ------------------------------------------------------------------ requesting suggestions
function schedule() {
  clearTimeout(debounce);
  debounce = setTimeout(request, cfg().get<number>("debounceMs") ?? 250);
}

async function request() {
  const ed = vscode.window.activeTextEditor;
  if (!ed || !cfg().get<boolean>("enabled") || !token || !serverUrl() || !isCode(ed.document)) return;
  if (pending) return;
  inflight?.abort();
  const ac = new AbortController();
  inflight = ac;
  const timeout = setTimeout(() => ac.abort(), 10_000);
  const doc = ed.document;
  const version = doc.version;
  const cur = ed.selection.active;
  const start = Math.max(0, cur.line - 30);
  try {
    const res = await api(
      "/v1/next_edit",
      {
        filepath: rel(doc),
        lines: lines(doc, start, cur.line + 31),
        lines_start: start,
        cursor_line: cur.line,
        cursor_char: cur.character,
        recent_edits: currentEdits(doc),
        context: contextSnippets(doc),
      },
      ac.signal,
    );
    if (res.model && res.model !== activeModel) { activeModel = res.model; refreshStatus(); }
    if (!res.id) return;
    // Discard if the user kept typing while we waited.
    const now = vscode.window.activeTextEditor;
    if (!now || now.document !== doc || doc.version !== version || !now.selection.active.isEqual(cur)) return;
    if (norm(regionText(doc, res.region_start, res.region_end)) !== norm(res.original)) return;
    show(now, res as Suggestion);
  } catch (err: any) {
    if (err?.name === "AbortError") return;
    if (String(err?.message) === "bad token") setStatus("$(key) Mine: bad token", "Run 'Mine: Set API Token'");
    else setStatus("$(warning) Mine: offline", String(err?.message ?? err));
  } finally {
    clearTimeout(timeout);
    if (inflight === ac) inflight = undefined;
  }
}

// ------------------------------------------------------------------ showing suggestions
function show(ed: vscode.TextEditor, s: Suggestion) {
  const doc = ed.document;
  const cur = ed.selection.active;
  const original = regionText(doc, s.region_start, s.region_end);
  // cursor offset inside the region text
  let off = 0;
  for (let i = s.region_start; i < cur.line; i++) off += doc.lineAt(i).text.length + 1;
  off += cur.character;
  const before = original.slice(0, off);
  const after = original.slice(off);
  const sug = s.suggestion;

  const isInsertion =
    cur.line >= s.region_start && cur.line < s.region_end &&
    sug.length > original.length && sug.startsWith(before) && sug.endsWith(after);

  if (isInsertion) {
    const insertion = sug.slice(before.length, sug.length - after.length);
    pending = { s, doc, kind: "inline", pos: cur, remainder: insertion, lineCount: doc.lineCount };
    vscode.commands.executeCommand("editor.action.inlineSuggest.trigger");
    return;
  }

  pending = { s, doc, kind: "edit", pos: cur, remainder: "", lineCount: doc.lineCount };
  drawEdit(ed, s);
  vscode.commands.executeCommand("setContext", "mine.editVisible", true);
}

function drawEdit(ed: vscode.TextEditor, s: Suggestion) {
  const parts = diffLines(s.original.endsWith("\n") ? s.original : s.original + "\n",
                          s.suggestion.endsWith("\n") ? s.suggestion : s.suggestion + "\n");
  const removed: vscode.Range[] = [];
  const added: vscode.DecorationOptions[] = [];
  let line = s.region_start;
  const hover = new vscode.MarkdownString();
  hover.appendMarkdown("**Mine** — press **Tab** to apply, **Esc** to dismiss\n");
  hover.appendCodeblock(s.suggestion, ed.document.languageId);

  for (let i = 0; i < parts.length; i++) {
    const p = parts[i];
    const n = p.count ?? p.value.split("\n").length - 1;
    if (p.removed) {
      for (let k = 0; k < n; k++) removed.push(new vscode.Range(line + k, 0, line + k, 0));
      const next = parts[i + 1];
      if (next?.added) {
        const newLines = next.value.replace(/\n$/, "").split("\n");
        newLines.forEach((t, k) => {
          const at = Math.min(line + Math.min(k, n - 1), ed.document.lineCount - 1);
          const extra = k === n - 1 && newLines.length > n ? `   (+${newLines.length - n} more lines)` : "";
          if (k < n) added.push({ range: lineEnd(ed, at), renderOptions: { after: { contentText: "→ " + t.trim() + extra } }, hoverMessage: hover });
        });
        i++;
      }
      line += n;
    } else if (p.added) {
      const newLines = p.value.replace(/\n$/, "").split("\n");
      const at = Math.max(s.region_start, Math.min(line - 1, ed.document.lineCount - 1));
      const more = newLines.length > 1 ? `   (+${newLines.length - 1} more lines)` : "";
      added.push({ range: lineEnd(ed, at), renderOptions: { after: { contentText: "+ " + newLines[0].trim() + more } }, hoverMessage: hover });
    } else {
      line += n;
    }
  }
  ed.setDecorations(removedDeco, removed.map((r) => ({ range: r, hoverMessage: hover })));
  ed.setDecorations(addedDeco, added);
  ed.setDecorations(regionDeco, [new vscode.Range(s.region_start, 0, Math.max(s.region_start, s.region_end - 1), 0)]);
  setStatus("$(sparkle) Mine: Tab to apply edit", "Esc to dismiss");
}

function lineEnd(ed: vscode.TextEditor, l: number) {
  const len = ed.document.lineAt(l).text.length;
  return new vscode.Range(l, len, l, len);
}

function clearEdit() {
  for (const ed of vscode.window.visibleTextEditors) {
    ed.setDecorations(removedDeco, []);
    ed.setDecorations(addedDeco, []);
    ed.setDecorations(regionDeco, []);
  }
  vscode.commands.executeCommand("setContext", "mine.editVisible", false);
  refreshStatus();
}

// ------------------------------------------------------------------ outcomes / feedback
function resolve(outcome: "accepted" | "rejected" | "ignored") {
  const p = pending;
  if (!p) return;
  pending = undefined;
  if (p.kind === "edit") clearEdit();
  if (resolved.has(p.s.id)) return;
  resolved.add(p.s.id);
  if (outcome === "accepted") {
    api("/v1/feedback", { id: p.s.id, outcome }).catch(() => {});
    return;
  }
  // Wait, then report what the user actually ended up writing in that region.
  const delay = (cfg().get<number>("feedbackDelaySeconds") ?? 20) * 1000;
  setTimeout(() => {
    let final: string | undefined;
    if (!p.doc.isClosed) {
      const delta = p.doc.lineCount - p.lineCount;
      final = regionText(p.doc, p.s.region_start, Math.max(p.s.region_start, p.s.region_end + delta));
    }
    api("/v1/feedback", { id: p.s.id, outcome, final }).catch(() => {});
  }, delay);
}

async function acceptEdit() {
  const p = pending;
  const ed = vscode.window.activeTextEditor;
  if (!p || p.kind !== "edit" || !ed || ed.document !== p.doc) return;
  const doc = p.doc;
  const endLine = Math.min(p.s.region_end, doc.lineCount) - 1;
  const range = new vscode.Range(p.s.region_start, 0, endLine, doc.lineAt(endLine).text.length);
  applying = true;
  try {
    await ed.edit((b) => b.replace(range, p.s.suggestion));
  } finally {
    applying = false;
  }
  // put the cursor at the end of the last changed line
  const newLines = p.s.suggestion.split("\n");
  const oldLines = p.s.original.split("\n");
  let last = newLines.length - 1;
  for (let k = newLines.length - 1, j = oldLines.length - 1; k >= 0 && j >= 0 && newLines[k] === oldLines[j]; k--, j--) last = k - 1;
  last = Math.max(0, last);
  const l = p.s.region_start + last;
  const pos = new vscode.Position(l, doc.lineAt(l).text.length);
  ed.selection = new vscode.Selection(pos, pos);
  resolve("accepted");
  schedule(); // chain: look for the next edit right away
}

// ------------------------------------------------------------------ inline (ghost text) provider
const inlineProvider: vscode.InlineCompletionItemProvider = {
  provideInlineCompletionItems(doc, pos) {
    const p = pending;
    if (!p || p.kind !== "inline" || p.doc !== doc || !p.pos.isEqual(pos) || !p.remainder) return undefined;
    const item = new vscode.InlineCompletionItem(p.remainder, new vscode.Range(pos, pos), {
      command: "mine._inlineAccepted",
      title: "accepted",
      arguments: [p.s.id],
    });
    return [item];
  },
};

// ------------------------------------------------------------------ events
function onChange(e: vscode.TextDocumentChangeEvent) {
  if (!isCode(e.document) || e.contentChanges.length === 0) return;
  trackChange(e);
  const p = pending;
  if (p && p.doc === e.document) {
    if (p.kind === "inline") {
      const c = e.contentChanges[0];
      const typedAlong =
        e.contentChanges.length === 1 && c.rangeLength === 0 && c.range.start.isEqual(p.pos) &&
        c.text.length > 0 && p.remainder.startsWith(c.text);
      if (typedAlong) {
        p.remainder = p.remainder.slice(c.text.length);
        p.pos = e.document.positionAt(e.document.offsetAt(c.range.start) + c.text.length);
        if (!p.remainder) resolve("accepted");
        else return; // keep showing the rest of the ghost text
      } else {
        resolve("ignored");
      }
    } else if (!applying) {
      resolve("ignored");
    }
  }
  if (applying) return;
  if (e.reason === vscode.TextDocumentChangeReason.Undo || e.reason === vscode.TextDocumentChangeReason.Redo) return;
  if (vscode.window.activeTextEditor?.document === e.document) schedule();
}

function onSelection(e: vscode.TextEditorSelectionChangeEvent) {
  const p = pending;
  if (!p || e.textEditor.document !== p.doc) return;
  if (e.kind === vscode.TextEditorSelectionChangeKind.Command || e.kind === undefined) return;
  const l = e.selections[0].active.line;
  if (p.kind === "inline" ? !e.selections[0].active.isEqual(p.pos) : l < p.s.region_start - 2 || l > p.s.region_end + 2) {
    resolve("ignored");
  }
}

// ------------------------------------------------------------------ activation
export async function activate(context: vscode.ExtensionContext) {
  ctx = context;
  token = await ctx.secrets.get("mine.token");
  status = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
  status.command = "mine.openDashboard";
  status.show();
  refreshStatus();

  for (const d of vscode.workspace.textDocuments) snapshots.set(d.uri.toString(), d.getText());
  noteVisit(vscode.window.activeTextEditor);

  context.subscriptions.push(
    status,
    vscode.languages.registerInlineCompletionItemProvider({ pattern: "**" }, inlineProvider),
    vscode.workspace.onDidChangeTextDocument(onChange),
    vscode.workspace.onDidOpenTextDocument((d) => snapshots.set(d.uri.toString(), d.getText())),
    vscode.workspace.onDidCloseTextDocument((d) => { snapshots.delete(d.uri.toString()); snapshots.delete(d.uri.toString() + "#burst"); }),
    vscode.window.onDidChangeTextEditorSelection(onSelection),
    vscode.window.onDidChangeActiveTextEditor((ed) => { if (pending) resolve("ignored"); noteVisit(ed); }),
    vscode.workspace.onDidChangeConfiguration((e) => { if (e.affectsConfiguration("mine")) refreshStatus(); }),

    vscode.commands.registerCommand("mine.acceptEdit", acceptEdit),
    vscode.commands.registerCommand("mine.rejectEdit", () => resolve("rejected")),
    vscode.commands.registerCommand("mine._inlineAccepted", (id: string) => {
      if (pending?.s.id === id) resolve("accepted");
      schedule();
    }),
    vscode.commands.registerCommand("mine.setToken", async () => {
      const t = await vscode.window.showInputBox({ prompt: "Paste your Mine API token", password: true, ignoreFocusOut: true });
      if (t === undefined) return;
      token = t.trim() || undefined;
      if (token) await ctx.secrets.store("mine.token", token);
      else await ctx.secrets.delete("mine.token");
      refreshStatus();
      checkConnection();
    }),
    vscode.commands.registerCommand("mine.setServerUrl", async () => {
      const u = await vscode.window.showInputBox({
        prompt: "Your Mine server URL (e.g. https://abc123-8000.proxy.runpod.net)",
        value: serverUrl(), ignoreFocusOut: true,
      });
      if (u === undefined) return;
      await cfg().update("serverUrl", u.trim(), vscode.ConfigurationTarget.Global);
      refreshStatus();
      checkConnection();
    }),
    vscode.commands.registerCommand("mine.toggle", async () => {
      await cfg().update("enabled", !cfg().get<boolean>("enabled"), vscode.ConfigurationTarget.Global);
      if (pending) resolve("ignored");
      refreshStatus();
    }),
    vscode.commands.registerCommand("mine.openDashboard", () => {
      if (!serverUrl()) return vscode.commands.executeCommand("mine.setServerUrl");
      if (!token) return vscode.commands.executeCommand("mine.setToken");
      vscode.env.openExternal(vscode.Uri.parse(serverUrl() + "/"));
    }),
    vscode.commands.registerCommand("mine.trainNow", async () => {
      try {
        const r = await api("/v1/train", {});
        vscode.window.showInformationMessage(r.started ? "Mine: training started — it'll switch over automatically if the new version is better." : `Mine: ${r.reason}`);
      } catch (e: any) {
        vscode.window.showErrorMessage(`Mine: ${e.message}`);
      }
    }),
  );
  checkConnection();
}

async function checkConnection() {
  if (!serverUrl() || !token) return;
  try {
    const s = await api("/v1/stats");
    activeModel = s.active_model;
    refreshStatus();
  } catch (e: any) {
    setStatus("$(warning) Mine: offline", String(e?.message ?? e));
  }
}

export function deactivate() {
  inflight?.abort();
}
