"""
BargainBuddy — Gradio Web UI
============================
Run with:  python app.py

The UI shows:
  - A live table of all discovered opportunities (deal price, estimate, discount)
  - A colour-coded agent log panel updated in real time
  - A 3-D scatter plot of the product vectorstore coloured by category
  - Auto-runs the pipeline every 5 minutes; click any row to push a manual alert
"""

import logging
import queue
import threading
import time

import gradio as gr
import plotly.graph_objects as go
from dotenv import load_dotenv

from framework import DealAgentFramework
from log_utils import reformat
from agents.url_scout_agent import URLScoutAgent

load_dotenv(override=True)


_vectorstore_ready = False


def _ensure_vectorstore():
    """Build ChromaDB in a background thread so startup is not blocked."""
    def _build():
        global _vectorstore_ready
        import chromadb
        client = chromadb.PersistentClient(path="products_vectorstore")
        collection = client.get_or_create_collection("products")
        if collection.count() == 0:
            print("Vectorstore is empty — running setup_vectorstore.py …")
            try:
                import setup_vectorstore
                setup_vectorstore.main()
                count = collection.count()
                print(f"Vectorstore ready — {count} items indexed.")
            except Exception as e:
                print(f"ERROR: Vectorstore setup failed: {e}")
                import traceback
                traceback.print_exc()
                return
        else:
            print(f"Vectorstore already has {collection.count()} items — skipping setup.")
        _vectorstore_ready = True

    t = threading.Thread(target=_build, daemon=True)
    t.start()


_ensure_vectorstore()


class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))


def html_for(log_data):
    output = "<br>".join(log_data[-18:])
    return f"""
    <div id="scrollContent" style="height: 400px; overflow-y: auto; border: 1px solid #ccc;
         background-color: #1e1e2e; padding: 10px; font-family: monospace; font-size: 13px;">
    {output}
    </div>
    """


def setup_logging(log_queue):
    handler = QueueHandler(log_queue)
    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %z",
    )
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class App:
    def __init__(self):
        self.agent_framework = None
        self.url_scout = None

    def get_agent_framework(self):
        if not self.agent_framework:
            self.agent_framework = DealAgentFramework()
        return self.agent_framework

    def get_url_scout(self):
        if not self.url_scout:
            ensemble = self.get_agent_framework().planner.ensemble if (
                self.agent_framework and self.agent_framework.planner
            ) else None
            if ensemble is None:
                self.get_agent_framework().init_agents_as_needed()
                ensemble = self.agent_framework.planner.ensemble
            self.url_scout = URLScoutAgent(ensemble)
        return self.url_scout

    def run(self):
        with gr.Blocks(title="BargainBuddy", fill_width=True) as ui:
            log_data = gr.State([])

            def table_for(opps):
                return [
                    [
                        opp.deal.product_description,
                        f"${opp.deal.price:.2f}",
                        f"${opp.estimate:.2f}",
                        f"${opp.discount:.2f}",
                        opp.deal.url,
                    ]
                    for opp in opps
                ]

            def update_output(log_data, log_queue, result_queue):
                initial_result = table_for(self.get_agent_framework().memory)
                final_result = None
                while True:
                    try:
                        message = log_queue.get_nowait()
                        log_data.append(reformat(message))
                        yield log_data, html_for(log_data), final_result or initial_result
                    except queue.Empty:
                        try:
                            final_result = result_queue.get_nowait()
                            yield log_data, html_for(log_data), final_result or initial_result
                        except queue.Empty:
                            if final_result is not None:
                                break
                            time.sleep(0.1)

            def get_initial_plot():
                fig = go.Figure()
                fig.update_layout(title="Loading vector DB...", height=400)
                return fig

            def get_plot():
                documents, vectors, colors = DealAgentFramework.get_plot_data(max_datapoints=800)
                if len(vectors) == 0:
                    fig = go.Figure()
                    fig.update_layout(
                        title="Vector DB is empty — run setup_vectorstore.py first",
                        height=400,
                    )
                    return fig
                fig = go.Figure(
                    data=[
                        go.Scatter3d(
                            x=vectors[:, 0],
                            y=vectors[:, 1],
                            z=vectors[:, 2],
                            mode="markers",
                            marker=dict(size=2, color=colors, opacity=0.7),
                        )
                    ]
                )
                fig.update_layout(
                    scene=dict(
                        xaxis_title="x",
                        yaxis_title="y",
                        zaxis_title="z",
                        aspectmode="manual",
                        aspectratio=dict(x=2.2, y=2.2, z=1),
                        camera=dict(eye=dict(x=1.6, y=1.6, z=0.8)),
                    ),
                    height=400,
                    margin=dict(r=5, b=1, l=5, t=2),
                )
                return fig

            def do_run():
                new_opportunities = self.get_agent_framework().run()
                return table_for(new_opportunities)

            def run_with_logging(initial_log_data):
                log_queue = queue.Queue()
                result_queue = queue.Queue()
                setup_logging(log_queue)

                def worker():
                    result = do_run()
                    result_queue.put(result)

                thread = threading.Thread(target=worker)
                thread.start()

                for log_data, output, final_result in update_output(
                    initial_log_data, log_queue, result_queue
                ):
                    yield log_data, output, final_result

            def do_select(selected_index: gr.SelectData):
                opportunities = self.get_agent_framework().memory
                row = selected_index.index[0]
                opportunity = opportunities[row]
                self.get_agent_framework().planner.messenger.alert(opportunity)

            def analyse_url(url: str, history: list):
                url = url.strip()
                if not url:
                    history.append({"role": "assistant", "content": "Please paste a product URL to analyse."})
                    return history, ""
                history.append({"role": "user", "content": url})
                history.append({"role": "assistant", "content": "Analysing… this may take 15–30 seconds."})
                yield history, ""
                try:
                    verdict = self.get_url_scout().analyse(url)
                except Exception as e:
                    verdict = f"Something went wrong: {e}"
                history[-1] = {"role": "assistant", "content": verdict}
                yield history, ""

            # ── Layout ───────────────────────────────────────────────────────
            with gr.Row():
                gr.Markdown(
                    '<div style="text-align:center; font-size:28px; font-weight:bold;">'
                    'BargainBuddy</div>'
                )
            with gr.Row():
                gr.Markdown(
                    '<div style="text-align:center; font-size:14px; color:#aaa;">'
                    'Autonomous multi-agent deal hunter powered by Groq — '
                    'finds online bargains and notifies you in real time.</div>'
                )
            with gr.Row():
                opportunities_dataframe = gr.Dataframe(
                    headers=["Deal Description", "Listed Price", "True Value", "Discount", "URL"],
                    wrap=True,
                    column_widths=[6, 1, 1, 1, 3],
                    row_count=10,
                    col_count=5,
                    max_height=400,
                )
            with gr.Row():
                with gr.Column(scale=1):
                    logs = gr.HTML()
                with gr.Column(scale=1):
                    plot = gr.Plot(value=get_plot(), show_label=False)

            # ── URL Deal Checker ─────────────────────────────────────────────
            with gr.Row():
                gr.Markdown(
                    '<div style="font-size:18px; font-weight:bold; margin-top:16px;">'
                    'Should I Buy This? — Paste a Product URL</div>'
                )
            with gr.Row():
                chatbot = gr.Chatbot(type="messages", height=340, show_label=False)
            with gr.Row():
                url_input = gr.Textbox(
                    placeholder="https://www.amazon.com/dp/...",
                    show_label=False,
                    scale=5,
                )
                analyse_btn = gr.Button("Analyse", variant="primary", scale=1)

            # Auto-run on load
            ui.load(
                run_with_logging,
                inputs=[log_data],
                outputs=[log_data, logs, opportunities_dataframe],
            )

            # Auto-refresh every 5 minutes
            timer = gr.Timer(value=300, active=True)
            timer.tick(
                run_with_logging,
                inputs=[log_data],
                outputs=[log_data, logs, opportunities_dataframe],
            )

            # Click a row → push that deal's alert
            opportunities_dataframe.select(do_select)

            # URL analyser
            analyse_btn.click(analyse_url, inputs=[url_input, chatbot], outputs=[chatbot, url_input])
            url_input.submit(analyse_url, inputs=[url_input, chatbot], outputs=[chatbot, url_input])

        ui.launch(server_name="0.0.0.0", share=False)


if __name__ == "__main__":
    App().run()
