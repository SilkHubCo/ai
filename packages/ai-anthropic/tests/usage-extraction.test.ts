import { beforeEach, describe, expect, it, vi } from 'vitest'
import { chat } from '@tanstack/ai'
import { AnthropicTextAdapter } from '../src/adapters/text'
import type { ChatMiddleware, StreamChunk } from '@tanstack/ai'

const mocks = vi.hoisted(() => {
  const betaMessagesCreate = vi.fn()

  const client = {
    beta: {
      messages: {
        create: betaMessagesCreate,
      },
    },
  }

  return { betaMessagesCreate, client }
})

vi.mock('@anthropic-ai/sdk', () => {
  const { client } = mocks

  class MockAnthropic {
    beta = client.beta

    constructor(_: { apiKey: string }) {}
  }

  return { default: MockAnthropic }
})

const createAdapter = () =>
  new AnthropicTextAdapter({ apiKey: 'test-key' }, 'claude-opus-4-1')

function createMockStream(
  chunks: Array<Record<string, unknown>>,
): AsyncIterable<Record<string, unknown>> {
  return {
    // eslint-disable-next-line @typescript-eslint/require-await
    async *[Symbol.asyncIterator]() {
      for (const chunk of chunks) {
        yield chunk
      }
    },
  }
}

async function collectMockStream(
  stream: AsyncIterable<Record<string, unknown>>,
  middleware?: ChatMiddleware,
): Promise<Array<StreamChunk>> {
  mocks.betaMessagesCreate.mockResolvedValueOnce(stream)

  const chunks: Array<StreamChunk> = []
  for await (const chunk of chat({
    adapter: createAdapter(),
    messages: [{ role: 'user', content: 'Hello' }],
    ...(middleware ? { middleware: [middleware] } : {}),
  })) {
    chunks.push(chunk)
  }
  return chunks
}

type UsageLifecycleEvent = 'RUN_FINISHED' | 'onUsage' | 'RUN_ERROR'

function createUsageLifecycleMiddleware(
  lifecycle: Array<UsageLifecycleEvent>,
): ChatMiddleware {
  return {
    name: 'usage-lifecycle',
    onChunk(_ctx, chunk) {
      if (chunk.type === 'RUN_FINISHED' || chunk.type === 'RUN_ERROR') {
        lifecycle.push(chunk.type)
      }
    },
    onUsage() {
      lifecycle.push('onUsage')
    },
  }
}

describe('Anthropic usage extraction', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('combines GCP Agent Platform message_start input usage with message_delta output usage', async () => {
    const mockStream = createMockStream([
      {
        type: 'message_start',
        message: {
          id: 'msg_123',
          type: 'message',
          role: 'assistant',
          content: [],
          model: 'claude-opus-4-1',
          usage: {
            input_tokens: 100,
            output_tokens: 0,
          },
        },
      },
      {
        type: 'content_block_start',
        index: 0,
        content_block: { type: 'text', text: '' },
      },
      {
        type: 'content_block_delta',
        index: 0,
        delta: { type: 'text_delta', text: 'Hello world' },
      },
      {
        type: 'message_delta',
        delta: { stop_reason: 'end_turn' },
        usage: {
          output_tokens: 50,
        },
      },
      {
        type: 'message_stop',
      },
    ])

    const chunks = await collectMockStream(mockStream)

    const doneChunk = chunks.find((c) => c.type === 'RUN_FINISHED')
    expect(doneChunk).toBeDefined()
    expect(doneChunk?.usage).toMatchObject({
      promptTokens: 100,
      completionTokens: 50,
      totalTokens: 150,
    })
  })

  it('preserves GCP Agent Platform message_start cache usage', async () => {
    const mockStream = createMockStream([
      {
        type: 'message_start',
        message: {
          id: 'msg_123',
          type: 'message',
          role: 'assistant',
          content: [],
          model: 'claude-opus-4-1',
          usage: {
            input_tokens: 100,
            output_tokens: 0,
            cache_creation_input_tokens: 50,
            cache_read_input_tokens: 25,
          },
        },
      },
      {
        type: 'content_block_start',
        index: 0,
        content_block: { type: 'text', text: '' },
      },
      {
        type: 'content_block_delta',
        index: 0,
        delta: { type: 'text_delta', text: 'Hello world' },
      },
      {
        type: 'message_delta',
        delta: { stop_reason: 'end_turn' },
        usage: {
          output_tokens: 50,
        },
      },
      {
        type: 'message_stop',
      },
    ])

    const chunks = await collectMockStream(mockStream)

    const doneChunk = chunks.find((c) => c.type === 'RUN_FINISHED')
    expect(doneChunk).toBeDefined()
    expect(doneChunk?.usage).toMatchObject({
      promptTokens: 175,
      completionTokens: 50,
      totalTokens: 225,
    })
    expect(doneChunk?.usage?.promptTokensDetails).toEqual({
      cacheWriteTokens: 50,
      cachedTokens: 25,
    })
    expect(
      (doneChunk?.usage?.promptTokens ?? 0) -
        (doneChunk?.usage?.promptTokensDetails?.cacheWriteTokens ?? 0) -
        (doneChunk?.usage?.promptTokensDetails?.cachedTokens ?? 0),
    ).toBe(100)
  })

  it('attaches GCP Agent Platform usage to a max_tokens error without finishing the run', async () => {
    const mockStream = createMockStream([
      {
        type: 'message_start',
        message: {
          id: 'msg_123',
          type: 'message',
          role: 'assistant',
          content: [],
          model: 'claude-opus-4-1',
          usage: {
            input_tokens: 100,
            output_tokens: 0,
            cache_creation_input_tokens: 40,
            cache_read_input_tokens: 25,
          },
        },
      },
      {
        type: 'content_block_start',
        index: 0,
        content_block: { type: 'text', text: '' },
      },
      {
        type: 'content_block_delta',
        index: 0,
        delta: { type: 'text_delta', text: 'Truncated response' },
      },
      {
        type: 'message_delta',
        delta: { stop_reason: 'max_tokens' },
        usage: {
          output_tokens: 50,
        },
      },
      {
        type: 'message_stop',
      },
    ])
    const lifecycle: Array<UsageLifecycleEvent> = []
    const chunks = await collectMockStream(
      mockStream,
      createUsageLifecycleMiddleware(lifecycle),
    )

    expect(lifecycle).toEqual(['RUN_ERROR'])
    expect(chunks.at(-1)).toMatchObject({
      type: 'RUN_ERROR',
      usage: {
        promptTokens: 165,
        completionTokens: 50,
        totalTokens: 215,
        promptTokensDetails: {
          cacheWriteTokens: 40,
          cachedTokens: 25,
        },
      },
    })
  })

  it('attaches known GCP Agent Platform usage to a stream error without finishing the run', async () => {
    const mockStream: AsyncIterable<Record<string, unknown>> = {
      async *[Symbol.asyncIterator]() {
        yield {
          type: 'message_start',
          message: {
            id: 'msg_123',
            type: 'message',
            role: 'assistant',
            content: [],
            model: 'claude-opus-4-1',
            usage: {
              input_tokens: 100,
              output_tokens: 0,
              cache_creation_input_tokens: 40,
              cache_read_input_tokens: 25,
            },
          },
        }
        throw Object.assign(new Error('stream failed'), {
          code: 'stream_failed',
        })
      },
    }
    const lifecycle: Array<UsageLifecycleEvent> = []
    const chunks = await collectMockStream(
      mockStream,
      createUsageLifecycleMiddleware(lifecycle),
    )

    expect(lifecycle).toEqual(['RUN_ERROR'])
    expect(chunks.at(-1)).toMatchObject({
      type: 'RUN_ERROR',
      code: 'stream_failed',
      usage: {
        promptTokens: 165,
        completionTokens: 0,
        totalTokens: 165,
        promptTokensDetails: {
          cacheWriteTokens: 40,
          cachedTokens: 25,
        },
      },
    })
  })

  it('extracts server tool use metrics', async () => {
    const mockStream = createMockStream([
      {
        type: 'message_start',
        message: {
          id: 'msg_123',
          type: 'message',
          role: 'assistant',
          content: [],
          model: 'claude-opus-4-1',
          usage: {
            input_tokens: 100,
            output_tokens: 0,
          },
        },
      },
      {
        type: 'content_block_start',
        index: 0,
        content_block: { type: 'text', text: '' },
      },
      {
        type: 'content_block_delta',
        index: 0,
        delta: { type: 'text_delta', text: 'Hello world' },
      },
      {
        type: 'message_delta',
        delta: { stop_reason: 'end_turn' },
        usage: {
          input_tokens: 100,
          output_tokens: 50,
          server_tool_use: {
            web_search_requests: 3,
            web_fetch_requests: 2,
          },
        },
      },
      {
        type: 'message_stop',
      },
    ])

    mocks.betaMessagesCreate.mockResolvedValueOnce(mockStream)

    const chunks: Array<StreamChunk> = []
    for await (const chunk of chat({
      adapter: createAdapter(),
      messages: [{ role: 'user', content: 'Hello' }],
    })) {
      chunks.push(chunk)
    }

    const doneChunk = chunks.find((c) => c.type === 'RUN_FINISHED')
    expect(doneChunk).toBeDefined()
    expect(doneChunk?.usage?.providerUsageDetails).toMatchObject({
      serverToolUse: {
        webSearchRequests: 3,
        webFetchRequests: 2,
      },
    })
  })

  it('handles response with no cache tokens', async () => {
    const mockStream = createMockStream([
      {
        type: 'message_start',
        message: {
          id: 'msg_123',
          type: 'message',
          role: 'assistant',
          content: [],
          model: 'claude-opus-4-1',
          usage: {
            input_tokens: 100,
            output_tokens: 0,
          },
        },
      },
      {
        type: 'content_block_start',
        index: 0,
        content_block: { type: 'text', text: '' },
      },
      {
        type: 'content_block_delta',
        index: 0,
        delta: { type: 'text_delta', text: 'Hello world' },
      },
      {
        type: 'message_delta',
        delta: { stop_reason: 'end_turn' },
        usage: {
          input_tokens: 100,
          output_tokens: 50,
        },
      },
      {
        type: 'message_stop',
      },
    ])

    mocks.betaMessagesCreate.mockResolvedValueOnce(mockStream)

    const chunks: Array<StreamChunk> = []
    for await (const chunk of chat({
      adapter: createAdapter(),
      messages: [{ role: 'user', content: 'Hello' }],
    })) {
      chunks.push(chunk)
    }

    const doneChunk = chunks.find((c) => c.type === 'RUN_FINISHED')
    expect(doneChunk).toBeDefined()
    // No cache tokens and no server tool use: the detail objects must be
    // omitted entirely rather than emitted as empty `{}` (matches every other
    // adapter's guarded behavior).
    expect(doneChunk?.usage?.promptTokensDetails).toBeUndefined()
    expect(doneChunk?.usage?.providerUsageDetails).toBeUndefined()
  })

  it('defaults missing output_tokens to 0 instead of NaN', async () => {
    const mockStream = createMockStream([
      {
        type: 'message_start',
        message: {
          id: 'msg_123',
          type: 'message',
          role: 'assistant',
          content: [],
          model: 'claude-opus-4-1',
          usage: {
            input_tokens: 100,
            output_tokens: 0,
          },
        },
      },
      {
        type: 'content_block_start',
        index: 0,
        content_block: { type: 'text', text: '' },
      },
      {
        type: 'content_block_delta',
        index: 0,
        delta: { type: 'text_delta', text: 'Hello world' },
      },
      {
        type: 'message_delta',
        delta: { stop_reason: 'end_turn' },
        // output_tokens intentionally omitted to exercise the `?? 0` guard.
        usage: {
          input_tokens: 100,
        },
      },
      {
        type: 'message_stop',
      },
    ])

    mocks.betaMessagesCreate.mockResolvedValueOnce(mockStream)

    const chunks: Array<StreamChunk> = []
    for await (const chunk of chat({
      adapter: createAdapter(),
      messages: [{ role: 'user', content: 'Hello' }],
    })) {
      chunks.push(chunk)
    }

    const doneChunk = chunks.find((c) => c.type === 'RUN_FINISHED')
    expect(doneChunk).toBeDefined()
    expect(doneChunk?.usage?.completionTokens).toBe(0)
    expect(doneChunk?.usage?.totalTokens).toBe(100)
    expect(Number.isNaN(doneChunk?.usage?.totalTokens)).toBe(false)
  })
})
