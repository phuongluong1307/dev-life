import {
  AlertCircle,
  Bot,
  Check,
  ChevronDown,
  ChevronRight,
  Code,
  Copy,
  Download,
  FileEdit,
  Folder,
  Loader2,
  RefreshCw,
  Search,
  Send,
  Square,
  Terminal,
  Wrench,
  X,
} from 'lucide-react'
import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from 'react'

// ── Types ─────────────────────────────────────────────────────────────────────

interface LlmProvider {
  id: string
  name: string
  provider: string
  models: { id: string; name: string }[]
}

export interface AIAgentSidebarProps {
  initialWorkspacePath?: string
  projectContext?: string
  onFilesChanged?: () => void
  onClose: () => void
  storageKey?: string
  codeFiles?: Record<string, string>
}

interface CodeSnippet {
  id: string
  file: string
  startLine: number
  endLine: number
}

interface ToolStep {
  id: string
  toolName: string
  toolArgs: Record<string, any>
  result?: string
  success?: boolean
  expanded: boolean
}

interface ChatMsg {
  id: string
  role: 'user' | 'assistant'
  content: string
  steps?: ToolStep[]
  durationMs?: number
  error?: string
  timestamp: number
  changedFiles?: string[]
}

// ── Constants ─────────────────────────────────────────────────────────────────

const FILE_MUTATING_TOOLS = new Set(['write_file', 'edit_file'])

const TOOL_ICONS: Record<string, React.ReactNode> = {
  read_file: <Folder size={11} />,
  write_file: <FileEdit size={11} />,
  edit_file: <FileEdit size={11} />,
  list_files: <Folder size={11} />,
  search_code: <Search size={11} />,
  run_command: <Terminal size={11} />,
}

const TOOL_STATUS: Record<string, string> = {
  read_file: 'Reading file',
  write_file: 'Writing file',
  edit_file: 'Editing file',
  list_files: 'Listing files',
  search_code: 'Searching code',
  run_command: 'Running command',
}

const SUGGESTED_PROMPTS = [
  'List all files in this project',
  'Explain the project structure',
  'Find all TODO comments in the code',
  'Show me the main entry point',
  'List all functions and their purposes',
  'Check for potential bugs or issues',
]

// ── Helpers ───────────────────────────────────────────────────────────────────

function toolLabel(name: string): string {
  return name.replace(/_/g, ' ')
}

function formatTime(ts: number): string {
  return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

function shortArg(_name: string, args: Record<string, any>): string {
  if (args.path) return args.path as string
  if (args.command) return (args.command as string).slice(0, 60)
  if (args.pattern) return `"${args.pattern}"`
  return ''
}

// ── Syntax highlighter (lightweight, no deps) ─────────────────────────────────

const JS_KEYWORDS = new Set([
  'const', 'let', 'var', 'function', 'return', 'if', 'else', 'for', 'while', 'do',
  'class', 'extends', 'import', 'export', 'from', 'default', 'new', 'this', 'typeof',
  'instanceof', 'null', 'undefined', 'true', 'false', 'async', 'await', 'try', 'catch',
  'finally', 'throw', 'in', 'of', 'break', 'continue', 'switch', 'case', 'type',
  'interface', 'enum', 'implements', 'readonly', 'static', 'public', 'private',
  'protected', 'abstract', 'as', 'satisfies',
])

const PY_KEYWORDS = new Set([
  'def', 'class', 'return', 'if', 'elif', 'else', 'for', 'while', 'import', 'from',
  'as', 'try', 'except', 'finally', 'raise', 'with', 'in', 'not', 'and', 'or', 'is',
  'True', 'False', 'None', 'lambda', 'pass', 'break', 'continue', 'yield', 'global',
  'nonlocal', 'del', 'assert',
])

type Token = { type: 'keyword' | 'string' | 'comment' | 'number' | 'plain'; value: string }

function tokenizeLine(line: string, lang: string): Token[] {
  const keywords = lang === 'python' || lang === 'py' ? PY_KEYWORDS : JS_KEYWORDS
  const commentChar = lang === 'python' || lang === 'py' ? '#' : '//'
  const tokens: Token[] = []
  let i = 0

  const push = (type: Token['type'], value: string) => {
    const last = tokens[tokens.length - 1]
    if (type === 'plain' && last?.type === 'plain') {
      last.value += value
    } else {
      tokens.push({ type, value })
    }
  }

  while (i < line.length) {
    // Line comment
    if (line.startsWith(commentChar, i)) {
      tokens.push({ type: 'comment', value: line.slice(i) })
      break
    }

    // String literals
    const q = line[i]
    if (q === '"' || q === "'" || q === '`') {
      let j = i + 1
      while (j < line.length) {
        if (line[j] === '\\') { j += 2; continue }
        if (line[j] === q) { j++; break }
        j++
      }
      tokens.push({ type: 'string', value: line.slice(i, j) })
      i = j
      continue
    }

    // Numbers
    if (/[0-9]/.test(line[i]) && (i === 0 || /\W/.test(line[i - 1]))) {
      let j = i
      while (j < line.length && /[0-9._xXa-fA-F]/.test(line[j])) j++
      tokens.push({ type: 'number', value: line.slice(i, j) })
      i = j
      continue
    }

    // Identifiers / keywords
    if (/[a-zA-Z_$]/.test(line[i])) {
      let j = i
      while (j < line.length && /[a-zA-Z0-9_$]/.test(line[j])) j++
      const word = line.slice(i, j)
      push(keywords.has(word) ? 'keyword' : 'plain', word)
      i = j
      continue
    }

    push('plain', line[i])
    i++
  }

  return tokens
}

const TOKEN_COLORS: Record<Token['type'], string> = {
  keyword: 'text-blue-400',
  string: 'text-orange-300',
  comment: 'text-zinc-500 italic',
  number: 'text-purple-400',
  plain: 'text-(--color-ink)',
}

function HighlightedCode({ code, lang }: { code: string; lang: string }) {
  const shouldHighlight = ['js', 'jsx', 'ts', 'tsx', 'javascript', 'typescript', 'python', 'py', 'json'].includes(lang)

  if (!shouldHighlight) {
    return <>{code}</>
  }

  return (
    <>
      {code.split('\n').map((line, li, arr) => (
        <span key={li}>
          {tokenizeLine(line, lang).map((tok, ti) => (
            <span key={ti} className={TOKEN_COLORS[tok.type]}>
              {tok.value}
            </span>
          ))}
          {li < arr.length - 1 && '\n'}
        </span>
      ))}
    </>
  )
}

// ── Inline markdown renderer ───────────────────────────────────────────────────

function renderInline(text: string): React.ReactNode[] {
  const pattern = /(\*\*(.+?)\*\*|`([^`]+)`)/gs
  const nodes: React.ReactNode[] = []
  let last = 0
  let match: RegExpExecArray | null
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > last) nodes.push(text.slice(last, match.index))
    if (match[0].startsWith('**')) {
      nodes.push(
        <strong key={match.index} className="font-semibold">
          {match[2]}
        </strong>,
      )
    } else {
      nodes.push(
        <code
          key={match.index}
          className="px-1 py-0.5 rounded bg-(--color-canvas) text-(--color-primary) font-mono text-[11px]"
        >
          {match[3]}
        </code>,
      )
    }
    last = match.index + match[0].length
  }
  if (last < text.length) nodes.push(text.slice(last))
  return nodes
}

// ── MarkdownView ──────────────────────────────────────────────────────────────

function MarkdownView({ text }: { text: string }) {
  const [copiedIdx, setCopiedIdx] = useState<number | null>(null)

  const copyCode = (code: string, idx: number) => {
    navigator.clipboard.writeText(code).then(() => {
      setCopiedIdx(idx)
      setTimeout(() => setCopiedIdx(null), 1500)
    })
  }

  const segments = text.split(/(```(?:[^\n]*)\n[\s\S]*?```)/g)
  let codeIdx = 0

  return (
    <>
      {segments.map((seg, i) => {
        if (seg.startsWith('```')) {
          const lang = (seg.match(/^```([^\n]*)/) ?? [])[1]?.trim() || ''
          const body = seg.replace(/^```[^\n]*\n/, '').replace(/```$/, '')
          const idx = codeIdx++
          return (
            <div key={i} className="relative group mt-2 mb-1">
              <pre className="p-3 pr-8 rounded-sm bg-(--color-canvas) border border-(--color-hairline) text-[11px] font-mono overflow-x-auto whitespace-pre">
                <HighlightedCode code={body} lang={lang} />
              </pre>
              <button
                type="button"
                onClick={() => copyCode(body, idx)}
                title="Copy code"
                className="absolute top-1.5 right-1.5 p-1 rounded opacity-0 group-hover:opacity-100 transition-opacity bg-(--color-canvas-soft) border border-(--color-hairline) text-(--color-mute) hover:text-(--color-ink) cursor-pointer"
              >
                {copiedIdx === idx ? <Check size={10} /> : <Copy size={10} />}
              </button>
            </div>
          )
        }
        const lines = seg.split('\n')
        return (
          <span key={i}>
            {lines.map((line, li) => (
              <span key={li}>
                {renderInline(line)}
                {li < lines.length - 1 && <br />}
              </span>
            ))}
          </span>
        )
      })}
    </>
  )
}

// ── ActivityBadge ─────────────────────────────────────────────────────────────

function ActivityBadge({ steps, hasText }: { steps: ToolStep[]; hasText: boolean }) {
  const activeTool = [...steps].reverse().find((s) => s.result === undefined)
  let label: string
  if (activeTool) {
    const base = TOOL_STATUS[activeTool.toolName] ?? toolLabel(activeTool.toolName)
    const hint = shortArg(activeTool.toolName, activeTool.toolArgs)
    label = hint ? `${base}: ${hint.split('/').pop() ?? hint}` : base
  } else if (!hasText) {
    label = 'Thinking'
  } else {
    label = 'Responding'
  }

  return (
    <span className="flex items-center gap-1 text-[11px] text-(--color-primary) font-mono">
      <span
        className="w-1.5 h-1.5 rounded-full bg-[var(--color-primary)] shrink-0"
        style={{ animation: 'thinking-bounce 1s ease-in-out infinite' }}
      />
      {label}…
    </span>
  )
}

function ThinkingDots() {
  return (
    <div className="flex items-end gap-[3px] px-3 py-2.5 rounded-md bg-(--color-canvas-soft) border border-(--color-hairline) w-fit">
      {[0, 150, 300].map((delay) => (
        <span
          key={delay}
          className="block w-1.5 h-1.5 rounded-full bg-[var(--color-mute)]"
          style={{
            animation: 'thinking-bounce 1s ease-in-out infinite',
            animationDelay: `${delay}ms`,
          }}
        />
      ))}
    </div>
  )
}

// ── ToolStepCard ──────────────────────────────────────────────────────────────

function ToolStepCard({ step, onToggle }: { step: ToolStep; onToggle: () => void }) {
  const pending = step.result === undefined
  const isRunCommand = step.toolName === 'run_command'
  const icon = TOOL_ICONS[step.toolName] ?? <Wrench size={11} />
  const hint = shortArg(step.toolName, step.toolArgs)

  return (
    <div
      className={`border rounded-sm overflow-hidden text-[11px] font-mono my-0.5 ${
        isRunCommand ? 'border-orange-900/50' : 'border-(--color-hairline)'
      }`}
    >
      <button
        type="button"
        onClick={onToggle}
        className={`w-full flex items-center gap-2 px-2.5 py-1.5 hover:bg-[#1a1a1a] cursor-pointer select-none ${
          isRunCommand ? 'bg-orange-950/20' : 'bg-(--color-canvas-soft)'
        }`}
      >
        <span
          className={
            pending
              ? 'text-(--color-mute)'
              : step.success
                ? 'text-(--color-primary)'
                : 'text-red-400'
          }
        >
          {pending ? <Loader2 size={11} className="animate-spin" /> : icon}
        </span>
        <span className={`font-semibold ${isRunCommand ? 'text-orange-400' : 'text-(--color-ink)'}`}>
          {toolLabel(step.toolName)}
        </span>
        {isRunCommand && !pending && (
          <span className="text-[9px] px-1 py-0.5 rounded bg-orange-900/30 text-orange-400 border border-orange-900/30 shrink-0">
            shell
          </span>
        )}
        {hint && (
          <span className="text-(--color-mute) truncate flex-1 text-left">{hint}</span>
        )}
        <span className="text-(--color-mute) shrink-0">
          {step.expanded ? <ChevronDown size={11} /> : <ChevronRight size={11} />}
        </span>
      </button>

      {step.expanded && (
        <div className="border-t border-(--color-hairline) bg-[#080808] p-2 max-h-48 overflow-y-auto">
          <div className="text-(--color-mute) mb-1.5">
            {Object.entries(step.toolArgs).map(([k, v]) => (
              <div key={k} className="flex gap-1.5">
                <span className="text-blue-400 shrink-0">{k}:</span>
                <span className="text-(--color-body) break-all whitespace-pre-wrap">
                  {typeof v === 'string' ? v : JSON.stringify(v)}
                </span>
              </div>
            ))}
          </div>
          {step.result !== undefined && (
            <div
              className={`mt-1 pt-1.5 border-t border-(--color-hairline) whitespace-pre-wrap break-all ${step.success ? 'text-(--color-body)' : 'text-red-400'}`}
            >
              {step.result}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ── ToolStepsGroup — collapsible step list ────────────────────────────────────

function ToolStepsGroup({
  steps,
  msgId,
  isStreaming,
  onToggleStep,
}: {
  steps: ToolStep[]
  msgId: string
  isStreaming: boolean
  onToggleStep: (stepId: string) => void
}) {
  const [collapsed, setCollapsed] = useState(false)
  if (steps.length === 0) return null

  const doneCount = steps.filter((s) => s.result !== undefined).length
  const label = isStreaming
    ? `${doneCount}/${steps.length} steps`
    : `${steps.length} step${steps.length > 1 ? 's' : ''}`

  return (
    <div className="w-full">
      <button
        type="button"
        onClick={() => setCollapsed((v) => !v)}
        className="flex items-center gap-1.5 mb-1 text-[10px] text-(--color-mute) hover:text-(--color-ink) cursor-pointer select-none transition-colors"
      >
        {collapsed ? <ChevronRight size={10} /> : <ChevronDown size={10} />}
        <span className="font-mono">{label}</span>
        {!isStreaming && steps.some((s) => FILE_MUTATING_TOOLS.has(s.toolName) && s.success) && (
          <span className="text-(--color-primary)">· files changed</span>
        )}
      </button>
      {!collapsed && steps.map((s) => (
        <ToolStepCard key={`${msgId}-${s.id}`} step={s} onToggle={() => onToggleStep(s.id)} />
      ))}
    </div>
  )
}

// ── MessageBubble ─────────────────────────────────────────────────────────────

function MessageBubble({
  msg,
  isStreaming,
  streamingText,
  onRetry,
}: {
  msg: ChatMsg
  isStreaming: boolean
  streamingText: string
  onRetry?: () => void
}) {
  const [copied, setCopied] = useState(false)
  const displayText = isStreaming ? streamingText : msg.content

  const handleCopy = () => {
    navigator.clipboard.writeText(msg.error || displayText).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    })
  }

  if (!displayText && !msg.error) return null

  return (
    <div className="relative group">
      <div
        className={`max-w-[92%] px-3 py-2 rounded-md text-[13px] leading-relaxed wrap-break-word ${
          msg.role === 'user'
            ? 'bg-gray-700 text-white ml-auto'
            : msg.error
              ? 'bg-red-950/30 border border-red-900/50 text-red-300'
              : 'bg-(--color-canvas-soft) border border-(--color-hairline) text-(--color-ink)'
        }`}
      >
        {msg.error ? (
          <div className="flex flex-col gap-2">
            <span className="flex items-center gap-1.5">
              <AlertCircle size={13} />
              {msg.error}
            </span>
            {onRetry && (
              <button
                type="button"
                onClick={onRetry}
                className="flex items-center gap-1 text-[11px] text-red-400 hover:text-red-300 cursor-pointer w-fit border border-red-900/40 rounded px-1.5 py-0.5 hover:border-red-700/50 transition-colors"
              >
                <RefreshCw size={10} />
                Retry
              </button>
            )}
          </div>
        ) : isStreaming ? (
          <>
            <MarkdownView text={streamingText} />
            <span className="inline-block w-1 h-3.5 bg-current ml-0.5 animate-pulse align-middle" />
          </>
        ) : msg.role === 'assistant' ? (
          <MarkdownView text={msg.content} />
        ) : (
          msg.content
        )}
      </div>

      {!isStreaming && !msg.error && displayText && (
        <button
          type="button"
          onClick={handleCopy}
          title="Copy"
          className={`absolute top-1.5 ${msg.role === 'user' ? '-left-7' : '-right-7'} p-1 rounded opacity-0 group-hover:opacity-100 transition-opacity bg-(--color-canvas-soft) border border-(--color-hairline) text-(--color-mute) hover:text-(--color-ink) cursor-pointer`}
        >
          {copied ? <Check size={11} /> : <Copy size={11} />}
        </button>
      )}
    </div>
  )
}

// ── ChangedFilesBadge ─────────────────────────────────────────────────────────

function ChangedFilesBadge({ files }: { files: string[] }) {
  const [expanded, setExpanded] = useState(false)
  if (files.length === 0) return null
  return (
    <div className="mt-1">
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="flex items-center gap-1 text-[10px] text-(--color-primary) font-mono cursor-pointer hover:underline"
      >
        <FileEdit size={10} />
        {files.length} file{files.length > 1 ? 's' : ''} modified
        {expanded ? <ChevronDown size={9} /> : <ChevronRight size={9} />}
      </button>
      {expanded && (
        <div className="mt-1 pl-3 flex flex-col gap-0.5">
          {files.map((f) => (
            <span key={f} className="text-[10px] font-mono text-(--color-mute)">
              {f}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

// ── ModelSelector ─────────────────────────────────────────────────────────────

function ModelSelector({
  providers,
  selectedProviderId,
  selectedModelId,
  onProviderChange,
  onModelChange,
}: {
  providers: LlmProvider[]
  selectedProviderId: string
  selectedModelId: string
  onProviderChange: (id: string) => void
  onModelChange: (id: string) => void
}) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const ref = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    function outside(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', outside)
    return () => document.removeEventListener('mousedown', outside)
  }, [])

  useEffect(() => {
    if (open) setTimeout(() => inputRef.current?.focus(), 40)
  }, [open])

  const currentProvider = providers.find((p) => p.id === selectedProviderId)
  const currentModel = currentProvider?.models.find((m) => m.id === selectedModelId)
  const filtered = (currentProvider?.models || []).filter((m) =>
    m.name.toLowerCase().includes(query.toLowerCase()),
  )

  return (
    <div className="flex items-center gap-1.5 flex-1 min-w-0">
      <select
        value={selectedProviderId}
        onChange={(e) => onProviderChange(e.target.value)}
        className="text-[11px] bg-(--color-canvas-soft) border border-(--color-hairline) text-(--color-ink) rounded-sm px-1.5 py-1 cursor-pointer outline-none"
      >
        {providers.map((p) => (
          <option key={p.id} value={p.id}>
            {p.name}
          </option>
        ))}
      </select>

      <div ref={ref} className="relative flex-1 min-w-0">
        <button
          type="button"
          onClick={() => setOpen(!open)}
          className="w-full flex items-center justify-between gap-1 text-[11px] bg-(--color-canvas-soft) border border-(--color-hairline) text-(--color-ink) rounded-sm px-2 py-1 cursor-pointer hover:border-(--color-primary) transition-colors"
        >
          <span className="truncate">{currentModel?.name || 'Select model'}</span>
          <ChevronDown size={10} className="shrink-0 text-(--color-mute)" />
        </button>

        {open && (
          <div className="absolute top-full left-0 right-0 mt-1 z-50 bg-(--color-canvas) border border-(--color-hairline) rounded-md shadow-xl overflow-hidden">
            <div className="p-1.5 border-b border-(--color-hairline)">
              <input
                ref={inputRef}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search models…"
                className="w-full text-[11px] bg-(--color-canvas-soft) border border-(--color-hairline) rounded-sm px-2 py-1 outline-none text-(--color-ink) placeholder:text-(--color-mute)"
              />
            </div>
            <div className="max-h-44 overflow-y-auto">
              {filtered.map((m) => (
                <button
                  key={m.id}
                  type="button"
                  onClick={() => { onModelChange(m.id); setOpen(false); setQuery('') }}
                  className={`w-full text-left text-[11px] px-3 py-1.5 hover:bg-(--color-canvas-soft) cursor-pointer ${m.id === selectedModelId ? 'text-(--color-primary)' : 'text-(--color-ink)'}`}
                >
                  {m.name}
                </button>
              ))}
              {filtered.length === 0 && (
                <div className="text-[11px] text-(--color-mute) px-3 py-2">No models</div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ── FileMentionPicker ─────────────────────────────────────────────────────────

function FileMentionPicker({
  query,
  files,
  onSelect,
  onClose,
}: {
  query: string
  files: string[]
  onSelect: (file: string) => void
  onClose: () => void
}) {
  const filtered = files
    .filter((f) => f.toLowerCase().includes(query.toLowerCase()))
    .slice(0, 8)
  const [activeIdx, setActiveIdx] = useState(0)

  useEffect(() => { setActiveIdx(0) }, [query])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'ArrowDown') { e.preventDefault(); setActiveIdx((i) => Math.min(i + 1, filtered.length - 1)) }
      if (e.key === 'ArrowUp') { e.preventDefault(); setActiveIdx((i) => Math.max(i - 1, 0)) }
      if (e.key === 'Enter') { e.preventDefault(); if (filtered[activeIdx]) onSelect(filtered[activeIdx]) }
      if (e.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [filtered, activeIdx, onSelect, onClose])

  if (filtered.length === 0) return null

  return (
    <div className="absolute bottom-full left-0 right-0 mb-1 z-50 bg-(--color-canvas) border border-(--color-hairline) rounded-md shadow-xl overflow-hidden">
      <div className="px-2 py-1 border-b border-(--color-hairline) text-[10px] text-(--color-mute) font-mono">
        @ file mention — ↑↓ navigate · Enter select · Esc close
      </div>
      {filtered.map((f, idx) => (
        <button
          key={f}
          type="button"
          onClick={() => onSelect(f)}
          className={`w-full text-left text-[11px] px-3 py-1.5 font-mono cursor-pointer transition-colors ${
            idx === activeIdx
              ? 'bg-(--color-primary)/10 text-(--color-primary)'
              : 'text-(--color-ink) hover:bg-(--color-canvas-soft)'
          }`}
        >
          {f}
        </button>
      ))}
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

export interface AIAgentSidebarHandle {
  addSnippet: (file: string, startLine: number, endLine: number) => void
}

const AIAgentSidebar = forwardRef<AIAgentSidebarHandle, AIAgentSidebarProps>(
  function AIAgentSidebar(
    { initialWorkspacePath = '', projectContext, onFilesChanged, onClose, storageKey, codeFiles },
    ref,
  ) {
    const lsKey = storageKey ? `ai-agent-model:${storageKey}` : null
    const chatLsKey = storageKey ? `ai-agent-chat:${storageKey}` : null

    // ── Provider / model ───────────────────────────────────────────────────────
    const [providers, setProviders] = useState<LlmProvider[]>([])
    const [selectedProviderId, setSelectedProviderId] = useState('')
    const [selectedModelId, setSelectedModelId] = useState('')
    const [loadingProviders, setLoadingProviders] = useState(true)

    // ── Workspace ──────────────────────────────────────────────────────────────
    const [workspacePath, setWorkspacePath] = useState(initialWorkspacePath)
    const [workspaceFiles, setWorkspaceFiles] = useState<string[]>([])

    // ── Chat ───────────────────────────────────────────────────────────────────
    const [messages, setMessages] = useState<ChatMsg[]>([])
    const [streamingMsgId, setStreamingMsgId] = useState<string | null>(null)
    const [streamingText, setStreamingText] = useState('')
    const [pendingSteps, setPendingSteps] = useState<ToolStep[]>([])
    const [isRunning, setIsRunning] = useState(false)
    const [input, setInput] = useState('')

    // ── @file mention ──────────────────────────────────────────────────────────
    const [mentionQuery, setMentionQuery] = useState('')
    const [showMentionPicker, setShowMentionPicker] = useState(false)

    // ── Snippet attachments ────────────────────────────────────────────────────
    const [codeSnippets, setCodeSnippets] = useState<CodeSnippet[]>([])
    const [showSnippetPicker, setShowSnippetPicker] = useState(false)
    const [snippetFile, setSnippetFile] = useState('')
    const [snippetStart, setSnippetStart] = useState('')
    const [snippetEnd, setSnippetEnd] = useState('')

    // ── Refs ───────────────────────────────────────────────────────────────────
    const requestIdRef = useRef<string | null>(null)
    const startTimeRef = useRef<number>(0)
    const cleanupRef = useRef<(() => void) | null>(null)
    const pendingStepsRef = useRef<ToolStep[]>([])
    const pendingChangedFilesRef = useRef<string[]>([])
    const isRunningRef = useRef(false)
    const bottomRef = useRef<HTMLDivElement>(null)
    const textareaRef = useRef<HTMLTextAreaElement>(null)
    const inputWrapRef = useRef<HTMLDivElement>(null)

    useImperativeHandle(ref, () => ({
      addSnippet: (file, startLine, endLine) => {
        setCodeSnippets((prev) => [...prev, { id: `snip-${Date.now()}`, file, startLine, endLine }])
        setShowSnippetPicker(false)
      },
    }), [])

    // ── Auto-focus on mount ────────────────────────────────────────────────────

    useEffect(() => {
      const t = setTimeout(() => textareaRef.current?.focus(), 80)
      return () => clearTimeout(t)
    }, [])

    // ── Load providers ─────────────────────────────────────────────────────────

    useEffect(() => {
      setLoadingProviders(true)
      window.api.listLlmProviders().then((res: any) => {
        if (res.success && res.providers?.length) {
          setProviders(res.providers)
          const saved = lsKey ? (() => { try { return JSON.parse(localStorage.getItem(lsKey) || 'null') } catch { return null } })() : null
          const savedProvider = saved?.providerId ? res.providers.find((p: any) => p.id === saved.providerId) : null
          const provider = savedProvider ?? res.providers[0]
          const model = savedProvider?.models.find((m: any) => m.id === saved?.modelId) ?? provider.models?.[0]
          setSelectedProviderId(provider.id)
          setSelectedModelId(model?.id || '')
        }
      }).finally(() => setLoadingProviders(false))
    }, [lsKey])

    // ── Persist model selection ────────────────────────────────────────────────

    useEffect(() => {
      if (!lsKey || !selectedProviderId || !selectedModelId) return
      localStorage.setItem(lsKey, JSON.stringify({ providerId: selectedProviderId, modelId: selectedModelId }))
    }, [lsKey, selectedProviderId, selectedModelId])

    // ── Load/save chat history ─────────────────────────────────────────────────

    useEffect(() => {
      if (!chatLsKey) return
      try {
        const saved = JSON.parse(localStorage.getItem(chatLsKey) || 'null')
        if (Array.isArray(saved) && saved.length > 0) setMessages(saved)
      } catch { /* ignore */ }
    }, [chatLsKey])

    // biome-ignore lint/correctness/useExhaustiveDependencies: persist on messages change
    useEffect(() => {
      if (!chatLsKey || messages.length === 0) return
      try {
        const toSave = messages
          .filter((m) => streamingMsgId === null || m.id !== streamingMsgId)
          .filter((m) => m.content || m.error)
          .slice(-60)
        localStorage.setItem(chatLsKey, JSON.stringify(toSave))
      } catch { /* ignore */ }
    }, [chatLsKey, messages, streamingMsgId])

    // ── Workspace sync + file listing ──────────────────────────────────────────

    useEffect(() => { setWorkspacePath(initialWorkspacePath) }, [initialWorkspacePath])

    useEffect(() => {
      if (!workspacePath) { setWorkspaceFiles([]); return }
      window.api.listWorkspaceFiles(workspacePath).then((res: any) => {
        if (res.success && res.files) setWorkspaceFiles(res.files)
      }).catch(() => {})
    }, [workspacePath])

    // ── Auto-scroll ────────────────────────────────────────────────────────────

    // biome-ignore lint/correctness/useExhaustiveDependencies: scroll on content change
    useEffect(() => {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [messages, pendingSteps, streamingText])

    // ── Provider helpers ───────────────────────────────────────────────────────

    const handleProviderChange = useCallback((id: string) => {
      setSelectedProviderId(id)
      const p = providers.find((x) => x.id === id)
      setSelectedModelId(p?.models?.[0]?.id || '')
    }, [providers])

    // ── Browse workspace ───────────────────────────────────────────────────────

    const handleBrowse = async () => {
      const res = await window.api.openDirectoryDialog()
      if (res.success && res.path) setWorkspacePath(res.path)
    }

    // ── Snippet helpers ────────────────────────────────────────────────────────

    const handleAddSnippet = useCallback(() => {
      const start = Number.parseInt(snippetStart)
      const end = Number.parseInt(snippetEnd)
      const file = snippetFile || Object.keys(codeFiles || {})[0] || ''
      if (!file || Number.isNaN(start) || Number.isNaN(end) || start < 1 || end < start) return
      setCodeSnippets((prev) => [...prev, { id: `snip-${Date.now()}`, file, startLine: start, endLine: end }])
      setSnippetStart('')
      setSnippetEnd('')
      setShowSnippetPicker(false)
    }, [snippetFile, snippetStart, snippetEnd, codeFiles])

    const handleRemoveSnippet = useCallback((id: string) => {
      setCodeSnippets((prev) => prev.filter((s) => s.id !== id))
    }, [])

    // ── @file mention ──────────────────────────────────────────────────────────

    const handleInputChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
      const val = e.target.value
      setInput(val)
      const cursor = e.target.selectionStart ?? val.length
      const before = val.slice(0, cursor)
      const atMatch = before.match(/@([\w./\-]*)$/)
      if (atMatch) {
        setMentionQuery(atMatch[1])
        setShowMentionPicker(true)
      } else {
        setShowMentionPicker(false)
      }
    }, [])

    const handleMentionSelect = useCallback((file: string) => {
      const cursor = textareaRef.current?.selectionStart ?? input.length
      const before = input.slice(0, cursor)
      const atMatch = before.match(/@([\w./\-]*)$/)
      if (!atMatch) return
      const start = cursor - atMatch[0].length
      setInput(input.slice(0, start) + `@${file}` + input.slice(cursor))
      setShowMentionPicker(false)
      setTimeout(() => textareaRef.current?.focus(), 0)
    }, [input])

    // ── Core agent send ────────────────────────────────────────────────────────

    const executeAgentMessage = useCallback(async (
      text: string,
      overrideHistory?: { role: 'user' | 'assistant'; content: string }[],
    ) => {
      if (isRunningRef.current || !selectedProviderId || !selectedModelId || !workspacePath) return

      // Resolve @file mentions → prepend file content blocks
      let agentMessage = text
      const mentionMatches = [...text.matchAll(/@([\w./\-]+)/g)]
      if (mentionMatches.length > 0) {
        const fileBlocks = await Promise.all(
          mentionMatches.map(async (m) => {
            const res = await window.api.readWorkspaceFile(workspacePath, m[1]).catch(() => null)
            if (!res?.success || !res.content) return null
            const ext = m[1].split('.').pop() || ''
            return `[File: ${m[1]}]\n\`\`\`${ext}\n${res.content}\n\`\`\``
          }),
        )
        const validBlocks = fileBlocks.filter(Boolean).join('\n\n')
        if (validBlocks) agentMessage = `${validBlocks}\n\n${text}`
      }

      // Append snippet attachments
      if (codeSnippets.length > 0 && codeFiles) {
        const blocks = codeSnippets.map((s) => {
          const lines = (codeFiles[s.file] || '').split('\n')
          const content = lines.slice(s.startLine - 1, s.endLine).join('\n')
          const lang = s.file.endsWith('.jsx') ? 'jsx' : 'js'
          return `[${s.file} — lines ${s.startLine}–${s.endLine}]\n\`\`\`${lang}\n${content}\n\`\`\``
        }).join('\n\n')
        agentMessage = `${blocks}\n\n${agentMessage}`
      }

      isRunningRef.current = true
      setInput('')
      setCodeSnippets([])
      setShowSnippetPicker(false)
      setShowMentionPicker(false)
      setIsRunning(true)
      startTimeRef.current = Date.now()

      const userMsgId = `msg-${Date.now()}-u`
      const assistantMsgId = `msg-${Date.now()}-a`

      setMessages((prev) => [
        ...prev,
        { id: userMsgId, role: 'user', content: text, timestamp: Date.now() },
        { id: assistantMsgId, role: 'assistant', content: '', steps: [], timestamp: Date.now() },
      ])
      setStreamingMsgId(assistantMsgId)
      setStreamingText('')
      pendingStepsRef.current = []
      pendingChangedFilesRef.current = []
      setPendingSteps([])

      const history = overrideHistory ?? messages
        .filter((m) => m.role === 'user' || (m.role === 'assistant' && m.content))
        .map((m) => ({ role: m.role, content: m.content }))

      cleanupRef.current?.()
      cleanupRef.current = null

      const requestId = crypto.randomUUID()
      requestIdRef.current = requestId

      const cleanup = window.api.onAgentEvent((ev: any) => {
        if (ev.requestId !== requestIdRef.current) return

        if (ev.type === 'token') {
          setStreamingText((prev) => prev + (ev.token || ''))
        } else if (ev.type === 'tool-start') {
          const step: ToolStep = {
            id: `step-${Date.now()}-${Math.random()}`,
            toolName: ev.toolName || '',
            toolArgs: ev.toolArgs || {},
            expanded: false,
          }
          pendingStepsRef.current = [...pendingStepsRef.current, step]
          setPendingSteps(pendingStepsRef.current)
        } else if (ev.type === 'tool-result') {
          if (ev.toolName && FILE_MUTATING_TOOLS.has(ev.toolName) && ev.toolSuccess) {
            const last = [...pendingStepsRef.current].reverse().find(
              (s) => s.toolName === ev.toolName && s.result === undefined,
            )
            if (last?.toolArgs?.path) {
              const fp = last.toolArgs.path as string
              if (!pendingChangedFilesRef.current.includes(fp)) {
                pendingChangedFilesRef.current = [...pendingChangedFilesRef.current, fp]
              }
            }
          }
          pendingStepsRef.current = pendingStepsRef.current.map((s) =>
            s.toolName === ev.toolName && s.result === undefined
              ? { ...s, result: ev.toolResult || '', success: ev.toolSuccess }
              : s,
          )
          setPendingSteps(pendingStepsRef.current)
        } else if (ev.type === 'done') {
          const finalSteps = pendingStepsRef.current
          const changedFiles = pendingChangedFilesRef.current
          if (ev.filesChanged && onFilesChanged) onFilesChanged()
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsgId
                ? { ...m, content: ev.fullText || '', steps: finalSteps, durationMs: Date.now() - startTimeRef.current, changedFiles }
                : m,
            ),
          )
          pendingStepsRef.current = []
          pendingChangedFilesRef.current = []
          setStreamingMsgId(null)
          setStreamingText('')
          setPendingSteps([])
          isRunningRef.current = false
          setIsRunning(false)
          cleanup()
          cleanupRef.current = null
        } else if (ev.type === 'error') {
          const finalSteps = pendingStepsRef.current
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsgId
                ? { ...m, content: '', error: ev.error || 'Unknown error', steps: finalSteps, durationMs: Date.now() - startTimeRef.current }
                : m,
            ),
          )
          pendingStepsRef.current = []
          pendingChangedFilesRef.current = []
          setStreamingMsgId(null)
          setStreamingText('')
          setPendingSteps([])
          isRunningRef.current = false
          setIsRunning(false)
          cleanup()
          cleanupRef.current = null
        }
      })
      cleanupRef.current = cleanup

      const res = await window.api.sendAgentMessage({
        requestId,
        providerId: selectedProviderId,
        modelId: selectedModelId,
        workspacePath,
        projectContext,
        conversationHistory: history,
        userMessage: agentMessage,
      })

      if (!res.success) {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsgId
              ? { ...m, error: res.error || 'Failed to start agent', durationMs: 0 }
              : m,
          ),
        )
        setStreamingMsgId(null)
        setIsRunning(false)
        cleanup()
        cleanupRef.current = null
      }
    }, [
      selectedProviderId, selectedModelId, workspacePath, projectContext,
      messages, onFilesChanged, codeSnippets, codeFiles,
    ])

    const handleSend = useCallback(async () => {
      const text = input.trim()
      if (!text) return
      await executeAgentMessage(text)
    }, [input, executeAgentMessage])

    const handleRetry = useCallback(async (assistantMsgId: string) => {
      const msgIdx = messages.findIndex((m) => m.id === assistantMsgId)
      if (msgIdx < 1) return
      const userMsg = messages[msgIdx - 1]
      if (userMsg.role !== 'user') return
      const historyBefore = messages
        .slice(0, msgIdx - 1)
        .filter((m) => m.role === 'user' || (m.role === 'assistant' && m.content))
        .map((m) => ({ role: m.role, content: m.content }))
      setMessages((prev) => prev.filter((m) => m.id !== assistantMsgId && m.id !== userMsg.id))
      await executeAgentMessage(userMsg.content, historyBefore)
    }, [messages, executeAgentMessage])

    const handleStop = async () => {
      if (requestIdRef.current) await window.api.cancelAgent(requestIdRef.current)
      cleanupRef.current?.()
      cleanupRef.current = null
      pendingStepsRef.current = []
      pendingChangedFilesRef.current = []
      setStreamingMsgId(null)
      setStreamingText('')
      setPendingSteps([])
      isRunningRef.current = false
      setIsRunning(false)
    }

    const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (showMentionPicker) return // let FileMentionPicker handle arrows/enter
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault()
        handleSend()
      }
    }

    // ── Toggle steps ───────────────────────────────────────────────────────────

    const toggleStep = useCallback((msgId: string, stepId: string) => {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === msgId
            ? { ...m, steps: m.steps?.map((s) => (s.id === stepId ? { ...s, expanded: !s.expanded } : s)) }
            : m,
        ),
      )
    }, [])

    const togglePendingStep = useCallback((stepId: string) => {
      setPendingSteps((prev) =>
        prev.map((s) => (s.id === stepId ? { ...s, expanded: !s.expanded } : s)),
      )
    }, [])

    // ── Clear chat ─────────────────────────────────────────────────────────────

    const clearChat = () => {
      if (isRunning) return
      setMessages([])
      if (chatLsKey) localStorage.removeItem(chatLsKey)
    }

    // ── Export chat as Markdown ────────────────────────────────────────────────

    const exportChat = () => {
      if (messages.length === 0) return
      const md = messages
        .filter((m) => m.content || m.error)
        .map((m) => {
          const role = m.role === 'user' ? '**You**' : '**Agent**'
          const time = formatTime(m.timestamp)
          const body = m.error ? `> Error: ${m.error}` : m.content
          return `${role} · ${time}\n\n${body}`
        })
        .join('\n\n---\n\n')

      const blob = new Blob([md], { type: 'text/markdown' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `chat-${new Date().toISOString().slice(0, 10)}.md`
      a.click()
      URL.revokeObjectURL(url)
    }

    const canSend = Boolean(input.trim() && selectedProviderId && selectedModelId && workspacePath && !isRunning)

    // ── Render ─────────────────────────────────────────────────────────────────

    return (
      <div className="flex flex-col h-full w-full bg-(--color-canvas) border-l border-(--color-hairline) text-(--color-ink)">
        <style>
          {'@keyframes thinking-bounce{0%,80%,100%{transform:translateY(0)}40%{transform:translateY(-5px)}}'}
        </style>

        {/* Header */}
        <div className="flex items-center gap-2 px-3 py-2 border-b border-(--color-hairline) shrink-0">
          <Bot size={15} className="text-(--color-primary) shrink-0" />
          <span className="font-semibold text-[13px] flex-1">AI Coding Agent</span>
          {messages.length > 0 && !isRunning && (
            <>
              <button
                type="button"
                onClick={exportChat}
                title="Export as Markdown"
                className="text-(--color-mute) hover:text-(--color-ink) cursor-pointer p-0.5 rounded hover:bg-(--color-canvas-soft)"
              >
                <Download size={13} />
              </button>
              <button
                type="button"
                onClick={clearChat}
                className="text-[11px] text-(--color-mute) hover:text-(--color-ink) cursor-pointer px-1.5 py-0.5 rounded hover:bg-(--color-canvas-soft)"
              >
                Clear
              </button>
            </>
          )}
          <button
            type="button"
            onClick={onClose}
            className="text-(--color-mute) hover:text-(--color-ink) cursor-pointer p-0.5 rounded hover:bg-(--color-canvas-soft)"
          >
            <X size={14} />
          </button>
        </div>

        {/* Workspace bar */}
        <div className="flex items-center gap-1.5 px-3 py-1.5 border-b border-(--color-hairline) bg-(--color-canvas-soft) shrink-0">
          <Folder size={11} className="text-(--color-mute) shrink-0" />
          <span
            className="flex-1 text-[11px] font-mono text-(--color-body) truncate"
            title={workspacePath || 'No workspace selected'}
          >
            {workspacePath || <span className="text-(--color-mute) italic">No workspace</span>}
          </span>
          <button
            type="button"
            onClick={handleBrowse}
            className="shrink-0 text-[11px] text-(--color-primary) hover:underline cursor-pointer"
          >
            Browse
          </button>
        </div>

        {/* Model selector */}
        <div className="px-3 py-1.5 border-b border-(--color-hairline) shrink-0">
          {loadingProviders ? (
            <div className="flex items-center gap-1.5 text-[11px] text-(--color-mute)">
              <Loader2 size={11} className="animate-spin" />
              Loading providers…
            </div>
          ) : providers.length === 0 ? (
            <p className="text-[11px] text-(--color-mute)">
              No LLM providers configured. Add one in Settings → LLM Providers.
            </p>
          ) : (
            <ModelSelector
              providers={providers}
              selectedProviderId={selectedProviderId}
              selectedModelId={selectedModelId}
              onProviderChange={handleProviderChange}
              onModelChange={setSelectedModelId}
            />
          )}
        </div>

        {/* Chat area */}
        <div className="flex-1 overflow-y-auto px-3 py-3 flex flex-col gap-3">
          {messages.length === 0 && !isRunning && (
            <div className="flex flex-col items-center justify-center h-full gap-3 text-center px-4">
              <Bot size={32} className="text-(--color-mute)" />
              <p className="text-[13px] text-(--color-mute) leading-relaxed">
                {workspacePath
                  ? 'Ask me to read, write, or modify files in your project.'
                  : 'Select a workspace directory, then ask me to help with your code.'}
              </p>
              {workspacePath && (
                <div className="flex flex-wrap gap-1.5 justify-center mt-1">
                  {SUGGESTED_PROMPTS.map((s) => (
                    <button
                      key={s}
                      type="button"
                      onClick={() => setInput(s)}
                      className="text-[11px] px-2 py-1 rounded-sm border border-(--color-hairline) bg-(--color-canvas-soft) text-(--color-body) hover:border-(--color-primary) hover:text-(--color-primary) cursor-pointer transition-colors"
                    >
                      {s}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

          {messages.map((msg) => {
            const isCurrentStreaming = msg.id === streamingMsgId
            const stepsToShow = isCurrentStreaming ? pendingSteps : (msg.steps ?? [])

            return (
              <div
                key={msg.id}
                className={`flex flex-col gap-1 ${msg.role === 'user' ? 'items-end' : 'items-start'}`}
              >
                {/* Role label + timestamp */}
                <div className={`flex items-center gap-1.5 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
                  <span className="text-[10px] font-semibold text-(--color-mute) uppercase tracking-wider">
                    {msg.role === 'user' ? 'You' : 'Agent'}
                  </span>
                  <span className="text-[10px] text-(--color-mute)">{formatTime(msg.timestamp)}</span>
                </div>

                {/* Tool steps — collapsible group */}
                {msg.role === 'assistant' && stepsToShow.length > 0 && (
                  <ToolStepsGroup
                    steps={stepsToShow}
                    msgId={msg.id}
                    isStreaming={isCurrentStreaming}
                    onToggleStep={(sid) =>
                      isCurrentStreaming ? togglePendingStep(sid) : toggleStep(msg.id, sid)
                    }
                  />
                )}

                {/* Thinking dots */}
                {isCurrentStreaming && !streamingText && pendingSteps.length === 0 && <ThinkingDots />}

                {/* Activity badge */}
                {isCurrentStreaming && pendingSteps.some((s) => s.result === undefined) && !streamingText && (
                  <ActivityBadge steps={pendingSteps} hasText={false} />
                )}

                {/* Message bubble */}
                <MessageBubble
                  msg={msg}
                  isStreaming={isCurrentStreaming}
                  streamingText={streamingText}
                  onRetry={msg.error ? () => handleRetry(msg.id) : undefined}
                />

                {/* Activity badge below text */}
                {isCurrentStreaming && streamingText && (
                  <ActivityBadge steps={pendingSteps} hasText={true} />
                )}

                {/* Changed files */}
                {msg.role === 'assistant' && !isCurrentStreaming && (msg.changedFiles?.length ?? 0) > 0 && (
                  <ChangedFilesBadge files={msg.changedFiles!} />
                )}

                {/* Duration */}
                {msg.durationMs !== undefined && msg.role === 'assistant' && (
                  <span className="text-[10px] text-(--color-mute) font-mono">
                    {(msg.durationMs / 1000).toFixed(1)}s
                  </span>
                )}
              </div>
            )
          })}

          <div ref={bottomRef} />
        </div>

        {/* Input bar */}
        <div className="border-t border-(--color-hairline) px-3 py-2 shrink-0">
          {/* Snippet picker */}
          {showSnippetPicker && codeFiles && Object.keys(codeFiles).length > 0 && (
            <div className="mb-2 flex items-center gap-1.5 p-2 rounded-sm bg-(--color-canvas-soft) border border-(--color-hairline)">
              <select
                value={snippetFile || Object.keys(codeFiles)[0]}
                onChange={(e) => setSnippetFile(e.target.value)}
                className="text-[11px] bg-(--color-canvas) border border-(--color-hairline) text-(--color-ink) rounded-sm px-1.5 py-1 outline-none cursor-pointer"
              >
                {Object.keys(codeFiles).map((f) => (
                  <option key={f} value={f}>{f}</option>
                ))}
              </select>
              <span className="text-[11px] text-(--color-mute)">lines</span>
              <input
                type="number"
                min={1}
                value={snippetStart}
                onChange={(e) => setSnippetStart(e.target.value)}
                placeholder="from"
                className="w-14 text-[11px] bg-(--color-canvas) border border-(--color-hairline) text-(--color-ink) rounded-sm px-1.5 py-1 outline-none [appearance:textfield]"
              />
              <span className="text-[11px] text-(--color-mute)">–</span>
              <input
                type="number"
                min={1}
                value={snippetEnd}
                onChange={(e) => setSnippetEnd(e.target.value)}
                placeholder="to"
                onKeyDown={(e) => e.key === 'Enter' && handleAddSnippet()}
                className="w-14 text-[11px] bg-(--color-canvas) border border-(--color-hairline) text-(--color-ink) rounded-sm px-1.5 py-1 outline-none [appearance:textfield]"
              />
              <button type="button" onClick={handleAddSnippet} className="text-[11px] px-2 py-1 rounded-sm bg-(--color-primary) text-black font-medium cursor-pointer hover:opacity-90">
                Add
              </button>
              <button type="button" onClick={() => setShowSnippetPicker(false)} className="ml-auto text-(--color-mute) hover:text-(--color-ink) cursor-pointer">
                <X size={12} />
              </button>
            </div>
          )}

          {/* Snippet chips */}
          {codeSnippets.length > 0 && (
            <div className="mb-1.5 flex flex-wrap gap-1">
              {codeSnippets.map((s) => (
                <span key={s.id} className="flex items-center gap-1 text-[11px] font-mono px-1.5 py-0.5 rounded-xs bg-(--color-primary)/10 border border-(--color-primary)/30 text-(--color-primary)">
                  {s.file.split('/')[0]}:{s.startLine}–{s.endLine}
                  <button type="button" onClick={() => handleRemoveSnippet(s.id)} className="cursor-pointer opacity-60 hover:opacity-100">
                    <X size={10} />
                  </button>
                </span>
              ))}
            </div>
          )}

          {/* Textarea + mention picker wrapper */}
          <div ref={inputWrapRef} className="relative flex items-end gap-2">
            {/* @file mention picker */}
            {showMentionPicker && workspaceFiles.length > 0 && (
              <FileMentionPicker
                query={mentionQuery}
                files={workspaceFiles}
                onSelect={handleMentionSelect}
                onClose={() => setShowMentionPicker(false)}
              />
            )}

            <textarea
              ref={textareaRef}
              value={input}
              onChange={handleInputChange}
              onKeyDown={handleKeyDown}
              placeholder={
                !workspacePath
                  ? 'Select a workspace first…'
                  : isRunning
                    ? 'Agent is running…'
                    : 'Ask anything… (Enter to send, @ to mention a file)'
              }
              disabled={isRunning || !workspacePath}
              rows={1}
              style={{ resize: 'none', minHeight: '60px', maxHeight: '120px' }}
              className="flex-1 bg-(--color-canvas-soft) border border-(--color-hairline) rounded-md px-3 py-2 text-[13px] text-(--color-ink) placeholder:text-(--color-mute) outline-none focus:border-(--color-primary) transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              onInput={(e) => {
                const t = e.currentTarget
                t.style.height = 'auto'
                t.style.height = `${Math.min(t.scrollHeight, 120)}px`
              }}
            />

            {isRunning ? (
              <button
                type="button"
                onClick={handleStop}
                title="Stop agent"
                className="shrink-0 flex items-center justify-center w-8 h-8 rounded-md bg-red-600 hover:bg-red-700 text-white cursor-pointer transition-colors"
              >
                <Square size={13} fill="currentColor" />
              </button>
            ) : (
              <button
                type="button"
                onClick={handleSend}
                disabled={!canSend}
                title="Send (Enter)"
                className="shrink-0 flex items-center justify-center w-8 h-8 rounded-md bg-(--color-primary) hover:opacity-90 text-black cursor-pointer transition-opacity disabled:opacity-40 disabled:cursor-not-allowed"
              >
                <Send size={13} />
              </button>
            )}
          </div>

          <div className="flex items-center justify-between mt-1">
            <p className="text-[10px] text-(--color-mute)">
              Enter · Shift+Enter newline · @ mention file
            </p>
            {codeFiles && Object.keys(codeFiles).length > 0 && (
              <button
                type="button"
                onClick={() => setShowSnippetPicker((v) => !v)}
                title="Attach code lines"
                className={`flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded-xs cursor-pointer transition-colors ${
                  showSnippetPicker || codeSnippets.length > 0
                    ? 'text-(--color-primary) bg-(--color-primary)/10'
                    : 'text-(--color-mute) hover:text-(--color-ink)'
                }`}
              >
                <Code size={11} />
                Add lines
              </button>
            )}
          </div>
        </div>
      </div>
    )
  },
)

export default AIAgentSidebar
