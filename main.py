"""
CLI Entry Point for Industry-Style Agentic RAG.
Supports interactive mode, single queries, running notebook demos, and document ingestion.
"""

import sys
import argparse
from config import check_api_keys
from graph import ask_agent, get_agent_app


DEMO_QUESTIONS = [
    "How does federated learning detect network intrusions in IoT or vehicular networks?",
    "What are the defense mechanisms against adversarial or collusion attacks in federated learning?",
    "Hello, how are you?",
    "What is Tavily Search and why is it useful for AI agents and RAG workflows?",
]


def run_demos():
    print("\nRunning Agentic RAG Demonstrations...")
    app = get_agent_app()
    for idx, q in enumerate(DEMO_QUESTIONS, 1):
        print(f"\n>>> DEMO {idx} of {len(DEMO_QUESTIONS)} <<<")
        ask_agent(q, app=app)


def interactive_mode():
    print("\n" + "=" * 70)
    print("Welcome to Industry-Style Agentic RAG (LangGraph + Groq + Pinecone + Tavily)")
    print("Type your question below, or type 'exit' or 'quit' to end session.")
    print("=" * 70)

    app = get_agent_app()

    while True:
        try:
            user_input = input("\nAsk a question > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break

            ask_agent(user_input, app=app)
        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting.")
            break
        except Exception as e:
            print(f"\n[Error during query processing]: {e}")


def visualize_graph():
    try:
        app = get_agent_app()
        mermaid_syntax = app.get_graph().draw_mermaid()
        print("\nMermaid Workflow Diagram:")
        print("```mermaid")
        print(mermaid_syntax)
        print("```")
        try:
            png_bytes = app.get_graph().draw_mermaid_png()
            with open("graph.png", "wb") as f:
                f.write(png_bytes)
            print("\nSaved workflow visualization image to 'graph.png'")
        except Exception as img_err:
            print(f"(Optional PNG render skipped: {img_err})")
    except Exception as e:
        print(f"Could not render graph: {e}")


def main():
    parser = argparse.ArgumentParser(description="Industry-Style Agentic RAG CLI")
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Run a single question query through the agentic graph"
    )
    parser.add_argument(
        "--demo", "-d",
        action="store_true",
        help="Run all 4 predefined demo queries from the notebook"
    )
    parser.add_argument(
        "--ingest", "-i",
        action="store_true",
        help="Run document ingestion & Pinecone index creation"
    )
    parser.add_argument(
        "--ui", "-u",
        action="store_true",
        help="Launch the interactive Streamlit Web UI"
    )

    args = parser.parse_args()

    # Launch Streamlit UI
    if args.ui:
        import subprocess
        print("Launching Streamlit Web UI on local browser...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
        return

    # Ingestion mode
    if args.ingest:
        from ingest import run_ingestion
        run_ingestion()
        return

    # Check required API keys before proceeding with graph execution
    check_api_keys()

    if args.visualize:
        visualize_graph()
        return

    if args.demo:
        run_demos()
        return

    if args.query:
        ask_agent(args.query)
        return

    # Default to interactive mode
    interactive_mode()


if __name__ == "__main__":
    main()
