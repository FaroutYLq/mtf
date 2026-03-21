# Browser GUI

```bash
pip install -e ".[gui]"
mtf-gui
```

Opens `http://localhost:8501`.

## Sidebar controls

- **Agent counts** — literature / fitting / reviewer parallelism
- **Debate rounds** — max synthesis iterations per phase
- **Physics domains** — GPD domain list (e.g. `condensed_matter`, `qft`, `nuclear`)
- **GPD toggle** — enable or disable physics verification servers
- **Models** — select Claude model per agent type

## Workflow

1. Enter your phenomenon description in the text area.
2. Optionally upload images (PNG, JPG, GIF, WebP) or data files.
3. Click **Run Analysis**.
4. Each phase (image digest → literature → fitting → review) appears as an expanding panel as it completes.
5. When MTF needs your approval or feedback it pauses and shows inline **Approve / Reject / Feedback** buttons — no terminal required.
6. The final report appears at the bottom and can be downloaded.

![MTF Streamlit GUI](https://i.imgur.com/ADiU2OX.png)

## Architecture note

The GUI runs the async orchestrator in a daemon thread with its own event loop. Two `queue.Queue` objects bridge the orchestrator thread to Streamlit's reactive rerun model: the orchestrator posts `("show"|"ask"|"confirm", payload, reply_q)` messages; Streamlit polls the queue on each rerun and posts answers back through `reply_q`. This means the GUI is fully reactive without any polling sleeps.
