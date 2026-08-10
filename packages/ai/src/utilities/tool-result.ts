import type { ContentPart } from '../types'

const CONTENT_PART_TYPES = new Set([
  'text',
  'image',
  'audio',
  'video',
  'document',
])

/**
 * Structural check for a single `ContentPart`. A text part must carry a string
 * `content`; every other modality must carry a `source` with `type` of
 * `'url' | 'data'` and a string `value`.
 */
export function isContentPart(value: unknown): value is ContentPart {
  if (typeof value !== 'object' || value === null) return false
  const part = value as Record<string, unknown>
  if (typeof part.type !== 'string' || !CONTENT_PART_TYPES.has(part.type)) {
    return false
  }
  if (part.type === 'text') {
    return typeof part.content === 'string'
  }
  const source = part.source
  if (typeof source !== 'object' || source === null) return false
  const src = source as Record<string, unknown>
  if (typeof src.value !== 'string') return false
  // `data` sources require a mimeType (matches ContentPartDataSource); `url`
  // sources don't. Requiring it here keeps the runtime guard consistent with
  // the type and avoids emitting `data:undefined;base64,...` downstream.
  if (src.type === 'data') return typeof src.mimeType === 'string'
  return src.type === 'url'
}

/**
 * True iff `value` is a NON-EMPTY array whose every element is a valid
 * `ContentPart`. Empty arrays and mixed arrays return false so they continue
 * to be treated as ordinary (stringified) data — this keeps the auto-detection
 * footgun narrow.
 */
export function isContentPartArray(
  value: unknown,
): value is Array<ContentPart> {
  return Array.isArray(value) && value.length > 0 && value.every(isContentPart)
}

/**
 * Normalize a tool's return value for transport:
 * - string            → unchanged
 * - ContentPart array → unchanged (multimodal, passed through to the adapter)
 * - anything else     → `JSON.stringify`
 */
export function normalizeToolResult(
  result: unknown,
): string | Array<ContentPart> {
  if (typeof result === 'string') return result
  if (isContentPartArray(result)) return result
  return JSON.stringify(result)
}

/**
 * Undo the wire stringification of a tool result: TOOL_CALL_END.result and
 * TOOL_CALL_RESULT.content are string-only per the AG-UI spec, so a multimodal
 * ContentPart array crosses the stream as JSON. Restore genuinely multimodal
 * arrays; collapse text-only arrays to their joined string (the legacy shape);
 * return anything else unchanged.
 */
export function restoreToolResultContent(
  wireContent: string,
): string | Array<ContentPart> {
  let parsed: unknown
  try {
    parsed = JSON.parse(wireContent)
  } catch {
    return wireContent
  }
  if (!isContentPartArray(parsed)) return wireContent
  if (parsed.some((part) => part.type !== 'text')) return parsed
  return parsed
    .map((part) => (part.type === 'text' ? part.content : ''))
    .join('')
}

/**
 * Split a wire tool result into the two shapes the processor writes: `content`
 * for the tool-result part (multimodal restored) and `output` for the
 * tool-call part (JSON parsed when the content is a plain string).
 */
export function parseToolResultWire(wireResult: string | Array<ContentPart>): {
  content: string | Array<ContentPart>
  output: unknown
} {
  const content = Array.isArray(wireResult)
    ? wireResult
    : restoreToolResultContent(wireResult)
  if (typeof content !== 'string') return { content, output: content }
  try {
    return { content, output: JSON.parse(content) }
  } catch {
    return { content, output: content }
  }
}
