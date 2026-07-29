# SilkHub core release

This branch is based on the `@tanstack/ai@0.39.1` tag and preserves
multimodal server-tool results when model messages are converted to UI
messages. Without this fix, image content parts returned by tools such as
`slidepool_search` are flattened to text before persistence and disappear
after reload.

The source package keeps its upstream `@tanstack/ai` name so the fork's
workspace, imports, and peer dependency resolution retain their normal
behavior. The release workflow changes only the packed package identity and
publishes it as `@silkhubco/ai`, allowing it to coexist with upstream. It also
attaches the package archive to the GitHub release so public consumers do not
need GitHub Packages credentials.

Release tags use `ai-v<package-version>-silkhubco.<release>`. The first core
release is `ai-v0.39.2-silkhubco.1`, and packages are versioned at `0.39.2`
so they remain compatible with consumers using the `^0.39.1` range.
