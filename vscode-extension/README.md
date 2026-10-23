# Kaptia VS Code context menu

Select code in a VS Code editor, right-click, then choose **問問 Kaptia**. Kaptia sends the selected code and a limited amount of surrounding text to its local-only service at `127.0.0.1:43871`. No source is sent to a cloud service.

This extension uses only Kaptia's own scratch checkpoint. It does not use an external model. The current bootstrap model is very small and may be unable to explain code reliably; the UI labels the answer as experimental and warns users to verify it.

## Current scope

- VS Code / Cursor-compatible VS Code extension host.
- Selected code + limited neighboring context.
- User-provided question and answer in a side panel.
- Explanation only: it does not edit files or apply code changes automatically.
