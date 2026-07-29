import { AnthropicVertex } from '@anthropic-ai/vertex-sdk'
import { chat } from '@tanstack/ai'
import { expect, it, vi } from 'vitest'
import { createAnthropicChatWithClient } from '../src'

it('routes an injected Vertex client request to the Vertex model endpoint', async () => {
  const requestedUrls: Array<string> = []
  const fetch = vi.fn((url: string | URL | Request) => {
    requestedUrls.push(String(url))
    return Promise.resolve(
      new Response('', {
        headers: { 'content-type': 'text/event-stream' },
        status: 200,
      }),
    )
  })
  const authClient = {
    projectId: 'test-project',
    getRequestHeaders: vi
      .fn()
      .mockResolvedValue(new Headers({ authorization: 'Bearer test-token' })),
  } as unknown as NonNullable<
    NonNullable<ConstructorParameters<typeof AnthropicVertex>[0]>['authClient']
  >
  const client = new AnthropicVertex({
    baseURL: 'https://aiplatform.eu.rep.googleapis.com/v1',
    region: 'eu',
    projectId: 'test-project',
    authClient,
    fetch,
    maxRetries: 0,
  })
  const adapter = createAnthropicChatWithClient('claude-opus-4-1', client)

  for await (const _ of chat({
    adapter,
    messages: [{ role: 'user', content: 'Hello' }],
  })) {
    // consume stream
  }

  expect(fetch).toHaveBeenCalledOnce()
  expect(requestedUrls).toEqual([
    'https://aiplatform.eu.rep.googleapis.com/v1/projects/test-project/locations/eu/publishers/anthropic/models/claude-opus-4-1:streamRawPredict',
  ])
})
