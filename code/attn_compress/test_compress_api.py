import json
import argparse
import textwrap
import sys
import urllib.request
import urllib.error
from typing import List, Dict, Any


def generate_dummy_data():
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]

    messages.append(
        {
            "role": "user",
            "content": (
                "Find the exact function that adds two numbers in calc.py, and output the complete function code.\n"
                "- Output ONLY the function definition (def ... + body).\n"
                "- Do NOT include any extra explanation.\n"
            ),
        }
    )

    noisy_tail = "".join(
        f"# other information {i:03d}: "
        "lorem ipsum dolor sit amet, consectetur adipiscing elit; "
        "sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.\n"
        for i in range(1, 5050)
    )

    calc_py_dump = (
        textwrap.dedent(
            '''\
            $ nl -ba calc.py | sed -n '1,220p'
                 1  """calc.py - toy calculator utilities.
                 2
                 3  This file is intentionally noisy for testing context compression.
                 4  The important part is the *exact* add() function below.
                 5  """
                 6
                 7  from __future__ import annotations
                 8
                 9  from dataclasses import dataclass
                10  from typing import Any
                11
                12  # --- lots of unrelated helpers ---
                13  def mul(a: int, b: int) -> int:
                14      return a * b
                15
                16  def sub(a: int, b: int) -> int:
                17      return a - b
                18
                19  def add(a: int, b: int) -> int:
                20      """Add two numbers and return the sum.
                21
                22      (KEY) This is the exact function requested.
                23      """
                24      return a + b
                25
                26  def div(a: int, b: int) -> float:
                27      if b == 0:
                28          raise ZeroDivisionError('b must not be zero')
                29      return a / b
                30
                31  # --- more irrelevant content omitted ---
            '''
        ).rstrip("\n")
        + "\n\n"
        + noisy_tail
    )

    messages.append(
        {
            "role": "assistant",
            "content": "Opening calc.py to locate the exact add() function.",
            "tool_calls": [
                {
                    "id": "call0",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps(
                            {"command": "nl -ba calc.py | sed -n '1,220p'"}
                        ),
                    },
                }
            ],
        }
    )

    messages.append(
        {
            "role": "tool",
            "content": calc_py_dump,
            "tool_call_id": "call0",
            "name": "bash",
        }
    )

    extract_dump = (
        textwrap.dedent(
            '''\
            $ python -c "<script omitted>"
            [INFO] Parsed calc.py successfully.
            [INFO] Found candidate: add(a, b) at lines 19-24.
            === EXTRACTED FUNCTION BEGIN ===
            def add(a: int, b: int) -> int:
                """Add two numbers and return the sum.

                (KEY) This is the exact function requested.
                """
                return a + b
            === EXTRACTED FUNCTION END ===
            '''
        ).rstrip("\n")
        + "\n\n"
        + noisy_tail
    )

    messages.append(
        {
            "role": "assistant",
            "content": "Extracting the add() function precisely to ensure we output the full definition.",
            "tool_calls": [
                {
                    "id": "call1",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps(
                            {"command": 'python -c "<script omitted>"'}
                        ),
                    },
                }
            ],
        }
    )

    messages.append(
        {
            "role": "tool",
            "content": extract_dump,
            "tool_call_id": "call1",
            "name": "bash",
        }
    )

    messages.append(
        {
            "role": "assistant",
            "content": "The add() function has been extracted successfully.",
        }
    )

    messages.append(
        {
            "role": "assistant",
            "content": "Outputting the final result.",
        }
    )

    messages.append(
        {
            "role": "tool",
            "content": 'def add(a: int, b: int) -> int:\n    """Add two numbers and return the sum.\n\n    (KEY) This is the exact function requested.\n    """\n    return a + b\n',
            "tool_call_id": None,
        }
    )

    step_indices =  [-1, -1, 0, 0, 1, 1, 2, 2, 2]

    return messages, step_indices


def main():
    parser = argparse.ArgumentParser(description="Test the compression API")
    parser.add_argument("--payload", type=str, help="Path to payload.json file")
    parser.add_argument(
        "--url", type=str, default="http://localhost:46405/compress", help="API URL"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/compressed_output.json",
        help="Output file path",
    )
    parser.add_argument(
        "--html",
        type=str,
        default="data/visualization.html",
        help="Output HTML visualization file path",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default="data/token_scores.png",
        help="Output token scores plot file path",
    )
    parser.add_argument(
        "--hist",
        type=str,
        default="data/token_scores_hist.png",
        help="Output token scores histogram file path",
    )
    args = parser.parse_args()

    payload = {
        "attn_ratio": 0.002,
        "attn_head": 0,
        "attn_tail": 0,
        "compress_tool_response": True,
        "compress_tool_call": True,
        "compress_assistant_content": False,
        "return_visualization": False,
        "chunking_method": "block", # "block", "message_block", "message_line"
        "hierarchical_message_ratio": 0.4,
        "randomize": 0,
        "block_split_method": "ppl", # "double_newline", "ppl"
        "selection_method": "knapsack", # "greedy", "knapsack"
        "ppl_spike_threshold_k": -2,
        "ppl_spike_method": "iqr", # "std", "robust_std", "iqr", "mad"
        "attn_layers": -1,
    }

    if args.payload:
        print(f"Loading payload from {args.payload}...")
        try:
            with open(args.payload, "r") as f:
                loaded_data = json.load(f)

            if isinstance(loaded_data, list):
                # Assume it's a list of messages
                payload["messages"] = loaded_data
            elif isinstance(loaded_data, dict):
                if "messages" in loaded_data:
                    # It's a full request object or close to it
                    payload.update(loaded_data)
                else:
                    # Assume it's a single message or something else, wrap it?
                    # Or maybe the user provided a dict that IS the messages list (not possible in JSON)
                    # Let's assume if it's a dict without "messages", it might be invalid or we treat it as part of messages?
                    # Safest is to assume it's a request body if it has keys like 'attn_ratio' etc.
                    payload.update(loaded_data)
                    if "messages" not in payload:
                        print(
                            "Error: Payload file must contain 'messages' list or be a list of messages."
                        )
                        return
        except Exception as e:
            print(f"Error reading payload file: {e}")
            return
    else:
        print("Generating dummy data...")
        payload["messages"], payload["step_indices"] = generate_dummy_data()
    print(
        f"Sending request to {args.url} with {len(payload.get('messages', []))} messages..."
    )
    print(f"Chunking method: {payload['chunking_method']}")
    if payload['chunking_method'] in ('message_block', 'message_line'):
        print(f"Hierarchical message ratio: {payload['hierarchical_message_ratio']}")

    req = urllib.request.Request(args.url, method="POST")
    req.add_header("Content-Type", "application/json")
    data = json.dumps(payload).encode("utf-8")

    try:
        with urllib.request.urlopen(req, data=data) as response:
            result = json.load(response)

        print("Compression successful!")
        stats = result.get("stats", {})
        print(f"Stats: {json.dumps(stats, indent=2)}")

        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Result saved to {args.output}")

        visualization_data = result.get("visualization")
        if visualization_data:
            html_content = visualization_data.get("html")
            if html_content:
                with open(args.html, "w", encoding="utf-8") as f:
                    f.write(html_content)
                print(f"Visualization saved to {args.html}")

            token_scores = visualization_data.get("token_scores")
            if token_scores:
                try:
                    import matplotlib.pyplot as plt

                    plt.figure(figsize=(12, 6))
                    plt.plot(token_scores, label="Token Scores", linewidth=0.5)
                    plt.xlabel("Token Index")
                    plt.ylabel("Attention Score")
                    plt.title("Token Attention Scores Distribution")
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(args.plot, dpi=300)
                    print(f"Token scores plot saved to {args.plot}")
                    plt.close()

                    # Histogram
                    plt.figure(figsize=(10, 6))
                    plt.hist(
                        token_scores,
                        bins=100,
                        color="skyblue",
                        edgecolor="black",
                        alpha=0.7,
                        log=True,
                    )
                    plt.xlabel("Attention Score")
                    plt.ylabel("Frequency (Log Scale)")
                    plt.title("Distribution of Attention Scores (Log Scale)")
                    plt.grid(True, alpha=0.3, which="both", linestyle="--")
                    plt.tight_layout()
                    plt.savefig(args.hist, dpi=300)
                    print(f"Token scores histogram saved to {args.hist}")
                    plt.close()
                except ImportError:
                    print("matplotlib not installed. Skipping plot generation.")
                except Exception as e:
                    print(f"Error generating plot: {e}")
        else:
            print("No visualization data returned.")

    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code} {e.reason}")
        try:
            error_body = e.read().decode("utf-8")
            print(f"Response content: {error_body}")
        except:
            pass
    except urllib.error.URLError as e:
        print(f"URL Error: {e.reason}")
        print(
            "Is the server running? (uvicorn attn_compress_service:app --host 0.0.0.0 --port 8000)"
        )
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
