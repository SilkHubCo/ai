import { test, expect } from './fixtures'
import {
  sendMessage,
  waitForResponse,
  getToolCalls,
  waitForAssistantText,
  featureUrl,
} from './helpers'
import { providersFor } from './test-matrix'

for (const provider of providersFor('tool-calling')) {
  test.describe(`${provider} — tool-calling`, () => {
    test('calls getGuitars and displays result', async ({
      page,
      testId,
      aimockPort,
    }) => {
      await page.goto(featureUrl(provider, 'tool-calling', testId, aimockPort))

      const chatResponse = page.waitForResponse('**/api/chat')
      await sendMessage(page, '[toolcall] what guitars do you have in stock')
      const response = await chatResponse
      await waitForResponse(page)

      const toolCalls = await getToolCalls(page)
      expect(toolCalls.length).toBeGreaterThanOrEqual(1)
      expect(toolCalls[0].name).toBe('getGuitars')

      // The parsed `input` is populated on the tool-call part (previously it
      // was always `undefined` at runtime — only `arguments` was set). It is
      // the raw `arguments` string parsed as JSON. `getGuitars` takes no
      // arguments, so it serializes as `{}` — assert the round-trip whenever
      // the args carry a JSON value (an empty/absent args string correctly
      // parses to no input, so we don't force a non-empty input there).
      const inputText = await page
        .getByTestId('tool-call-input-getGuitars')
        .locator('code')
        .first()
        .innerText()
      const argsText = (
        await page
          .getByTestId('tool-call-getGuitars')
          .locator('code')
          .first()
          .innerText()
      ).trim()
      if (argsText.length > 0) {
        expect(JSON.parse(inputText)).toEqual(JSON.parse(argsText))
      }

      // Wait for the text response after tool execution (agentic loop's second LLM call)
      await waitForAssistantText(page, 'Fender Stratocaster')

      if (provider === 'openai') {
        const events: Array<unknown> = (await response.text())
          .split('\n')
          .filter((line) => line.startsWith('data: '))
          .map((line) => JSON.parse(line.slice(6)))
        for (const type of [
          'TOOL_CALL_START',
          'TOOL_CALL_ARGS',
          'TOOL_CALL_END',
        ]) {
          expect(events).toContainEqual(
            expect.objectContaining({ type, toolCallId: 'call_guitars_stock' }),
          )
        }
      }
    })
  })
}
