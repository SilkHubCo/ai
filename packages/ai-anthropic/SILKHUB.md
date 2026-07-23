# SilkHub release branch

This branch is based on the `@tanstack/ai-anthropic@0.16.0` tag and backports
the preconfigured Messages-client seam from
[TanStack/ai#989](https://github.com/TanStack/ai/pull/989).

The source package keeps its upstream `@tanstack/ai-anthropic` name so the
fork's workspace, docs, and tests retain their normal resolution behavior.
The release workflow changes only the packed package identity and publishes it
as `@silkhubco/ai-anthropic`, allowing it to coexist with upstream. It also
attaches the package archive to the GitHub release so public consumers don't
need GitHub Packages credentials. Release tags use
`ai-anthropic-v<package-version>`. The fork's `main` branch remains an
unmodified upstream mirror.
