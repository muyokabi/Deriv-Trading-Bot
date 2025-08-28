# main.py
import asyncio
import websockets
import json
import random
from datetime import datetime
from rich.live import Live
from rich.table import Table
from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from rich.align import Align
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError
import time
import subprocess
import os
import numpy as np
import logging

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("main_bot.log")
                    ])
for handler in logging.root.handlers[:]:
    if isinstance(handler, logging.StreamHandler):
        logging.root.removeHandler(handler)

# --- CONFIG ---
APP_ID = 85473
API_TOKEN = "YUa7FW6khNwyW"  # replace with your Api token
SYMBOL = "1HZ10V"
STAKE = 0.35
TRADE_DURATION_SECONDS = 1 # This is the "duration" for the Deriv API call in ticks - CHANGED TO 1 TICK

# --- NEW CONFIG FOR REVAMP ---
MIN_CONFIDENCE_THRESHOLD = 0.51
DAILY_LOSS_LIMIT = 3.00
DAILY_PROFIT_TARGET = 5.00
UI_REFRESH_INTERVAL = 0.1

# --- STATE ---
account_balance = 0.0
trade_history = []
total_profit = 0.0
daily_profit = 0.0
total_trades = 0
last_tick = "0.00"
previous_tick = "0.00"
console = Console()
message_queue = asyncio.Queue()
trade_in_progress_message = "Waiting for next trade..."
current_status_message = "Initializing..."
model_ready = False
model_actively_predicting = False
model_ws_connection = None
model_accuracy_display = "N/A"
trading_active = True
last_ui_update_time = time.time()

# Global Rich Layout object
main_layout = Layout(name="root")

def make_initial_layout():
    main_layout.split(
        Layout(name="header", size=3),
        Layout(name="main_body", ratio=1),
        Layout(name="footer", size=1)
    )
    main_layout["main_body"].split_row(
        Layout(name="left_column", ratio=1),
        Layout(name="right_column", ratio=1)
    )
    main_layout["left_column"].split(
        Layout(name="stats_panel", size=15),
        Layout(name="trade_history_panel", ratio=1)
    )
    main_layout["right_column"].split(
        Layout(name="status_panel", size=9),
        Layout(name="info_log_panel", ratio=1),
        Layout(name="progress_panel", size=3)
    )

    main_layout["header"].update(Panel(Align.center(Text("🤖 Deriv Digit Bot", style="bold magenta"))))
    main_layout["footer"].update(Align.center(Text(f"App ID: {APP_ID} | Symbol: {SYMBOL}", style="dim white")))
    main_layout["stats_panel"].update(Panel("Loading stats...", border_style="bright_cyan"))
    main_layout["trade_history_panel"].update(Panel("Loading trade history...", border_style="yellow"))
    main_layout["status_panel"].update(Panel("Awaiting connection...", border_style="dark_orange"))
    main_layout["info_log_panel"].update(Panel(Text("No activity yet.", style="dim"), title="Activity Log", border_style="purple"))
    main_layout["progress_panel"].update(Panel("No active operations.", border_style="green"))

def build_stats_panel():
    global last_tick, previous_tick
    tick_color = "white"
    try:
        current_tick_float = float(last_tick)
        previous_tick_float = float(previous_tick)
        if current_tick_float > previous_tick_float:
            tick_color = "green"
        elif current_tick_float < previous_tick_float:
            tick_color = "red"
    except ValueError:
        pass

    stats_table = Table.grid(expand=True)
    stats_table.add_column(style="bold white")
    stats_table.add_column(justify="right", style="bold")

    stats_table.add_row("💱 Account Balance", str(f"[green]${account_balance:.2f}[/green]"))
    stats_table.add_row("")
    stats_table.add_row("💹 Total PnL", str(f"{'[green]' if total_profit >= 0 else '[red]'}${total_profit:.2f}[/]"))
    stats_table.add_row("📈 Daily PnL", str(f"{'[green]' if daily_profit >= 0 else '[red]'}${daily_profit:.2f}[/]"))
    stats_table.add_row("")
    stats_table.add_row("🥶 Total Trades", str(total_trades))
    stats_table.add_row("")
    stats_table.add_row("🏹 Current Tick", str(f"[{tick_color}]{last_tick}[/{tick_color}]"))
    stats_table.add_row("")
    stats_table.add_row("⌚ Time", str(datetime.now().strftime("%H:%M:%S")))

    return Panel(stats_table, title="📊 Bot Statistics", title_align="left", border_style="bright_cyan", height=15)

def build_trade_history_panel():
    trades = Table(
        title="📜 Last 17 Trades",
        title_style="bold yellow",
        border_style="green",
        show_header=True,
        header_style="bold underline",
        expand=True,
        show_lines=False,
        box=None
    )
    trades.add_column("⌚ Time", style="dim white", width=8, min_width=8)
    trades.add_column("🩻 Pred", style="bold cyan", width=5, min_width=5)
    trades.add_column("🤯 Dig", style="magenta", width=4, min_width=4)
    trades.add_column("📈 Entry", style="blue", width=6, min_width=6)
    trades.add_column("📉 Outcome", style="blue", width=7, min_width=7)
    trades.add_column("🙆 Res", style="white", width=5, min_width=5)
    trades.add_column("💷 P/L", justify="right", style="bold", width=6, min_width=6)

    for t in reversed(trade_history[-17:]):
        emoji = "🟢" if t['win'] else "🔴"
        pl_color = "[green]" if t['win'] else "[red]"

        profit_for_display = 0.0
        try:
            profit_for_display = float(t['profit'])
        except (ValueError, TypeError):
            logging.warning(f"Invalid profit value in trade history: {t['profit']}. Defaulting to 0.0.")
            profit_for_display = 0.0

        profit_string_with_color = f"{pl_color}{profit_for_display:+.2f}[/]"

        result_display = "N/A"
        trade_result_raw = t.get('result')
        if isinstance(trade_result_raw, str) and len(trade_result_raw) > 0:
            result_display = f"{emoji} {trade_result_raw[0]}"
        else:
            logging.warning(f"Trade history 'result' field is not a valid string or is empty: {trade_result_raw}. Displaying as 'N/A'.")
            result_display = f"{emoji} ?"

        trades.add_row(
            str(t["time"]),
            str(t['prediction']),
            str(t["barrier_digit"]),
            str(t["entry"]),
            str(t["outcome"]),
            result_display,
            profit_string_with_color
        )
    return Panel(trades, border_style="yellow", title_align="left")

def build_status_panel():
    global model_ready, model_actively_predicting, model_accuracy_display, trading_active

    model_status_text_content = ""
    if not model_ready:
        model_status_text_content = "[yellow]Initializing...[/yellow]"
    elif model_ready and not model_actively_predicting:
        spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        frame_index = int(time.time() * 10) % len(spinner_frames)
        model_status_text_content = f"[cyan]{spinner_frames[frame_index]} Awaiting first prediction...[/cyan]"
    else:
        model_status_text_content = "[green]Ready[/green]"

    model_status_text = Text.from_markup(model_status_text_content)

    trading_status_text = Text.from_markup("[green]Active[/green]") if trading_active else Text.from_markup("[bold red]Daily Limit Hit![/bold red]")

    accuracy_text_content = ""
    if isinstance(model_accuracy_display, float):
        accuracy_text_content = f"[green]{model_accuracy_display:.2%}[/green]"
    else:
        accuracy_text_content = f"[yellow]{model_accuracy_display}[/yellow]"
    accuracy_text = Text.from_markup(accuracy_text_content)

    return Panel(
        Align.center(
            Text.assemble(
                ("Current Status: ", "white bold"),
                (current_status_message, "green bold"),
                "\n\n",
                ("Trade in progress: ", "white bold"),
                (trade_in_progress_message, "cyan italic"),
                "\n\n",
                ("Model Status: ", "white bold"),
                model_status_text,
                "\n\n",
                ("Model Accuracy: ", "white bold"),
                accuracy_text,
                "\n\n",
                ("Trading Status: ", "white bold"),
                trading_status_text
            ),
            vertical="middle",
        ),
        title="🚦 Bot Status",
        title_align="left",
        border_style="dark_orange",
        height=9
    )

log_messages = []
def add_log_message(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_messages.append(Text(f"[{timestamp}] ") + Text.from_markup(message))
    if len(log_messages) > 9:
        log_messages.pop(0)
    logging.info(f"[UI_LOG] {message}")

def build_info_log_panel():
    log_text = Text("\n").join(log_messages)
    return Panel(log_text, title="📢 Activity Log", title_align="left", border_style="purple")

global_progress = Progress(
    SpinnerColumn("dots", style="bold cyan"),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(bar_width=None, style="cyan", complete_style="green"),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    console=console,
    refresh_per_second=10
)

def get_last_digit(price_str):
    try:
        return int(str(price_str).split('.')[-1][-1])
    except (ValueError, IndexError):
        logging.warning(f"Could not extract last digit from price string: '{price_str}'. Returning 0.")
        return 0

async def send(ws, data):
    try:
        await ws.send(json.dumps(data))
    except Exception as e:
        add_log_message(f"[bold red]Error sending data to WS: {e}[/bold red]")
        raise

async def receiver(ws):
    global last_tick, previous_tick
    try:
        async for message in ws:
            data = json.loads(message)
            if data.get("msg_type") == "tick":
                previous_tick = last_tick
                last_tick = data['tick']['quote']
            await message_queue.put(data)
    except ConnectionClosedOK:
        add_log_message("[yellow]WebSocket connection closed by server.[/yellow]")
    except ConnectionClosedError as e:
        add_log_message(f"[bold red]WebSocket connection closed unexpectedly: {e}[/bold red]")
    except asyncio.CancelledError:
        add_log_message("[yellow]Receiver task cancelled.[/yellow]")
    except Exception as e:
        add_log_message(f"[bold red]Error in receiver: {e}[/bold red]")

async def wait_for_message(msg_type, progress_task_id=None, timeout=30):
    try:
        while True:
            data = await asyncio.wait_for(message_queue.get(), timeout=timeout)
            if data.get("msg_type") == msg_type:
                if progress_task_id is not None and progress_task_id in global_progress.tasks:
                    if not global_progress.tasks[progress_task_id].finished:
                        global_progress.update(progress_task_id, visible=False)
                return data
            if "error" in data:
                add_log_message(f"[bold red]API Error: {data['error']['message']}[/bold red]")
                continue
    except asyncio.TimeoutError:
        add_log_message(f"[bold red]Timeout waiting for message type '{msg_type}'.[/bold red]")
        return {"error": {"message": f"Timeout waiting for {msg_type}"}}
    except asyncio.CancelledError:
        raise
    except Exception as e:
        add_log_message(f"[bold red]Error in wait_for_message: {e}[/bold red]")
        return {"error": {"message": f"Exception in wait_for_message: {e}"}}

MODEL_COMM_PORT = 8765
MODEL_READY_FILE = "model_ready.flag"

async def start_model_engine():
    global model_ready
    add_log_message("[bold blue]Starting model.py engine in background...[/bold blue]")
    if os.path.exists(MODEL_READY_FILE):
        os.remove(MODEL_READY_FILE)

    model_process = subprocess.Popen(["python", "model.py"],
                                     stdout=subprocess.DEVNULL, # Suppress stdout
                                     stderr=subprocess.DEVNULL) # Suppress stderr

    add_log_message("[yellow]Waiting for model engine to initialize...[/yellow]")
    model_init_task_id = global_progress.add_task("[yellow]Waiting for model to initialize...", total=480)

    progress_step = 0
    MAX_WAIT_STEPS = 480 # 4 minutes (480 * 0.5s)

    while not os.path.exists(MODEL_READY_FILE):
        if model_process.poll() is not None:
            add_log_message(f"[bold red]Model process terminated unexpectedly with exit code {model_process.returncode}.[/bold red]")
            global_progress.remove_task(model_init_task_id)
            return None
        if not global_progress.tasks[model_init_task_id].finished:
            global_progress.update(model_init_task_id, advance=1)
        await asyncio.sleep(0.5)
        progress_step += 1
        if progress_step > MAX_WAIT_STEPS:
            add_log_message("[bold red]Model initialization timed out![/bold red]")
            model_process.terminate()
            break

    if os.path.exists(MODEL_READY_FILE):
        global_progress.remove_task(model_init_task_id)
        model_ready = True
        add_log_message("[bold green]Model engine is ready![/bold green]")
    else:
        add_log_message("[bold red]Model engine did not become ready or timed out.[/bold red]")
        model_process = None

    return model_process

async def connect_to_model_ws():
    global model_ws_connection
    if model_ws_connection and model_ws_connection.open:
        return model_ws_connection
    try:
        model_ws_connection = await websockets.connect(f"ws://localhost:{MODEL_COMM_PORT}", open_timeout=5)
        add_log_message("✅ [green]Connected to model.py prediction server.[/green]")
        return model_ws_connection
    except ConnectionRefusedError:
        add_log_message("[bold red]Could not connect to model.py server. Model might not be running or ready.[/bold red]")
    except asyncio.TimeoutError:
        add_log_message("[bold red]Timeout connecting to model.py server.[/bold red]")
    except Exception as e:
        add_log_message(f"[bold red]Error connecting to model.py WS: {e}[/bold red]")
    return None

async def get_model_prediction():
    global model_ws_connection, model_accuracy_display, model_actively_predicting

    if not model_ws_connection or not model_ws_connection.open:
        add_log_message("[yellow]Model WS connection lost or not established. Attempting to reconnect...[/yellow]")
        model_ws_connection = await connect_to_model_ws()
        if not model_ws_connection:
            model_accuracy_display = "N/A"
            model_actively_predicting = False
            return -1, 0.0, 0.0, 0, 9, {} # Return dummy values
    try:
        await model_ws_connection.send(json.dumps({"request": "prediction"}))
        # Add a short timeout to prevent indefinite waiting if model.py is stuck
        response = await asyncio.wait_for(model_ws_connection.recv(), timeout=2) # 2-second timeout as tick interval is 2s
        data = json.loads(response)

        if all(key in data for key in ["prediction", "confidence", "accuracy", "barrier_over", "barrier_under"]):
            received_probabilities = data.get("probabilities", {})
            probabilities = {}
            if isinstance(received_probabilities, dict):
                for digit_str, prob_val in received_probabilities.items():
                    try:
                        digit = int(digit_str)
                        if 0 <= digit <= 9 and isinstance(prob_val, (int, float)):
                            probabilities[digit] = float(prob_val)
                        else:
                            add_log_message(f"[yellow]Skipping invalid probability entry: digit='{digit_str}', prob='{prob_val}'. Digit not 0-9 or prob not number.[/yellow]")
                    except ValueError:
                        add_log_message(f"[yellow]Skipping non-integer probability key: '{digit_str}'.[/yellow]")
                    except TypeError:
                        add_log_message(f"[yellow]Skipping invalid probability value type for key '{digit_str}': '{prob_val}'.[/yellow]")
            else:
                add_log_message(f"[bold red]Model returned probabilities in unexpected format (not a dictionary): {received_probabilities}. Treating as empty.[/bold red]")

            model_accuracy_display = data["accuracy"]
            model_actively_predicting = True

            return data["prediction"], data["confidence"], data["accuracy"], data["barrier_over"], data["barrier_under"], probabilities
        else:
            add_log_message(f"[red]Invalid prediction response from model: {data}. Missing expected keys.[/red]")
            model_accuracy_display = "Error"
            model_actively_predicting = False
            return -1, 0.0, 0.0, 0, 9, {}
    except asyncio.TimeoutError:
        add_log_message("[bold red]Timeout waiting for model prediction response.[/bold red]")
        model_ws_connection = None
        model_accuracy_display = "Timeout"
        model_actively_predicting = False
        return -1, 0.0, 0.0, 0, 9, {}
    except ConnectionClosedError as e:
        add_log_message(f"[bold yellow]Model WS connection closed during prediction: {e}. Will attempt reconnect.[/bold yellow]")
        model_ws_connection = None
        model_accuracy_display = "Disconnected"
        model_actively_predicting = False
        return -1, 0.0, 0.0, 0, 9, {}
    except Exception as e:
        add_log_message(f"[bold red]Error fetching prediction over persistent WS: {e}[/bold red]")
        model_ws_connection = None
        model_accuracy_display = "Error"
        model_actively_predicting = False
        return -1, 0.0, 0.0, 0, 9, {}

async def run_bot():
    global account_balance, total_profit, daily_profit, total_trades, trade_in_progress_message, current_status_message, model_ready, model_actively_predicting, model_ws_connection, model_accuracy_display, trading_active, last_ui_update_time

    make_initial_layout()

    ws = None
    receiver_task = None
    model_process = None

    with Live(main_layout, refresh_per_second=1000, screen=True, console=console) as live:
        main_layout["progress_panel"].update(global_progress)

        current_status_message = "Starting bot..."
        main_layout["stats_panel"].update(build_stats_panel())
        main_layout["status_panel"].update(build_status_panel())
        main_layout["info_log_panel"].update(build_info_log_panel())
        main_layout["trade_history_panel"].update(build_trade_history_panel())
        await asyncio.sleep(0.5)

        model_process = await start_model_engine()
        if not model_ready or model_process is None:
            current_status_message = "Model failed to initialize. Exiting."
            add_log_message("[bold red]Model not ready, cannot proceed with trading.[/bold red]")
            if model_process:
                try:
                    model_process.terminate()
                    await asyncio.sleep(0.5)
                except ProcessLookupError:
                    pass
            return

        model_ws_connection = await connect_to_model_ws()
        if not model_ws_connection:
            current_status_message = "Failed to connect to model. Exiting."
            add_log_message("[bold red]Could not establish persistent connection to model.py.[/bold red]")
            if model_process:
                try:
                    model_process.terminate()
                    await asyncio.sleep(0.5)
                except ProcessLookupError:
                    pass
            return

        try:
            current_status_message = "Connecting to Deriv WebSocket..."

            async with websockets.connect(f"wss://ws.binaryws.com/websockets/v3?app_id={APP_ID}") as connected_ws:
                ws = connected_ws
                receiver_task = asyncio.create_task(receiver(ws))

                auth_task_id = global_progress.add_task("[yellow]Authorizing with API...", total=100)
                global_progress.start_task(auth_task_id)
                add_log_message("[yellow]Sending authorization request...[/yellow]")
                await send(ws, {"authorize": API_TOKEN})

                for _ in range(5):
                    global_progress.update(auth_task_id, advance=20)
                    await asyncio.sleep(0.1)

                auth_response = await wait_for_message("authorize", auth_task_id)
                global_progress.remove_task(auth_task_id)

                if "error" in auth_response:
                    current_status_message = "Authentication Failed!"
                    add_log_message(f"❌ [red]Auth Error: {auth_response['error']['message']}[/red]")
                    return

                account_balance = float(auth_response["authorize"]["balance"])
                current_status_message = "Authorized & Connected!"
                add_log_message(f"✅ [green]Authorization successful. Balance: ${account_balance:.2f}[/green]")
                await asyncio.sleep(0.5)

                current_status_message = "Subscribing to market ticks..."
                add_log_message(f"📈 [yellow]Subscribing to ticks for [cyan]{SYMBOL}[/cyan]...[/yellow]")
                await send(ws, {"ticks": SYMBOL, "subscribe": 1})
                await asyncio.sleep(1)

                current_status_message = "Bot is active, awaiting next trade opportunity."

                while True:
                    if time.time() - last_ui_update_time >= UI_REFRESH_INTERVAL:
                        main_layout["stats_panel"].update(build_stats_panel())
                        main_layout["status_panel"].update(build_status_panel())
                        main_layout["info_log_panel"].update(build_info_log_panel())
                        main_layout["trade_history_panel"].update(build_trade_history_panel())
                        live.refresh()
                        last_ui_update_time = time.time()

                    if not trading_active:
                        current_status_message = "Daily limits reached. Trading paused."
                        add_log_message(f"[bold magenta]Daily trading limits reached. Profit: ${daily_profit:.2f}. Pausing.[/bold magenta]")
                        await asyncio.sleep(60)
                        continue

                    # Wait for a new tick from Deriv - This is the entry tick for the trade decision
                    entry_tick_data = await wait_for_message("tick")
                    entry_tick = entry_tick_data['tick']['quote']
                    entry_tick_digit = get_last_digit(entry_tick)
                    
                    trade_in_progress_message = "Getting latest prediction from model..."

                    # Immediately fetch the *latest* pre-calculated prediction from model.py
                    predicted_digit, confidence, current_model_accuracy, barrier_over, barrier_under, probabilities = await get_model_prediction()
                    model_accuracy_display = current_model_accuracy

                    if predicted_digit == -1 or not model_actively_predicting:
                        add_log_message("[bold yellow]Model not actively predicting or prediction failed. Skipping trade.[/bold yellow]")
                        if model_process and model_process.poll() is not None:
                            add_log_message("[bold red]Model process terminated unexpectedly. Shutting down bot.[/bold red]")
                            break
                        await asyncio.sleep(0.1) # Short sleep
                        continue

                    if confidence < MIN_CONFIDENCE_THRESHOLD:
                        add_log_message(f"[yellow]Model prediction confidence too low ({confidence:.2%}). Skipping trade based on general confidence filter.[/yellow]")
                        trade_in_progress_message = "Confidence too low."
                        await asyncio.sleep(0.1) # Short sleep
                        continue

                    prediction_type = None
                    digit_barrier = None

                    if predicted_digit >= 5:
                        prediction_type = "Over"
                        digit_barrier = 2
                    elif predicted_digit < 5:
                        prediction_type = "Under"
                        digit_barrier = 7

                    if not prediction_type or digit_barrier is None:
                        add_log_message("[bold red]Error in determining trade type/barrier based on prediction. Skipping trade.[/bold red]")
                        trade_in_progress_message = "Trade decision error."
                        await asyncio.sleep(0.1) # Short sleep
                        continue

                    add_log_message(f"🔮 Model Pred: {predicted_digit} (Conf: {confidence:.2%}). Trading [cyan]{prediction_type}[/cyan] [magenta]{digit_barrier}[/magenta].")

                    account_balance -= STAKE
                    add_log_message(f"💰 [blue]Executing trade: [bold cyan]{prediction_type}[/bold cyan] [magenta]{digit_barrier}[/magenta] for [red]-${STAKE:.2f}[/red]. New balance: ${account_balance:.2f}[/blue]")
                    trade_in_progress_message = f"Deducting ${STAKE:.2f} for trade..."

                    proposal_req = {
                        "proposal": 1,
                        "amount": STAKE,
                        "basis": "stake",
                        "contract_type": f"DIGIT{prediction_type.upper()}",
                        "currency": "USD",
                        "duration": TRADE_DURATION_SECONDS, # Now 1 tick
                        "duration_unit": "t",
                        "symbol": SYMBOL,
                        "barrier": str(digit_barrier)
                    }

                    trade_in_progress_message = "Sending proposal..."
                    proposal_task_id = global_progress.add_task("[yellow]Sending proposal...", total=100)
                    global_progress.start_task(proposal_task_id)
                    await send(ws, proposal_req)

                    for _ in range(5):
                        global_progress.update(proposal_task_id, advance=20)
                        await asyncio.sleep(0.02) # Faster update to match the 2-second cycle

                    proposal = await wait_for_message("proposal", proposal_task_id)
                    global_progress.remove_task(proposal_task_id)

                    if "error" in proposal:
                        trade_in_progress_message = "Proposal failed."
                        add_log_message(f"❌ [red]Proposal Error: {proposal['error']['message']}. Refunding stake.[/red]")
                        account_balance += STAKE
                        await asyncio.sleep(0.1)
                        continue

                    trade_in_progress_message = "Executing trade..."
                    buy_task_id = global_progress.add_task(f"[green]Buying contract for ${STAKE:.2f}...", total=100)
                    global_progress.start_task(buy_task_id)
                    add_log_message(f"💰 [green]Buying contract for [green]${STAKE:.2f}[/green]...[/green]")
                    await send(ws, {"buy": proposal["proposal"]["id"], "price": STAKE})

                    for _ in range(5):
                        global_progress.update(buy_task_id, advance=20)
                        await asyncio.sleep(0.02) # Faster update

                    buy_response = await wait_for_message("buy", buy_task_id)
                    global_progress.remove_task(buy_task_id)

                    if "error" in buy_response:
                        trade_in_progress_message = "Buy order failed."
                        add_log_message(f"❌ [red]Buy Error: {buy_response['error']['message']}. Refunding stake.[/red]")
                        account_balance += STAKE
                        await asyncio.sleep(0.1)
                        continue

                    contract_id = buy_response['buy']['contract_id']
                    add_log_message(f"✅ [green]Contract purchased. ID: {contract_id}[/green]")

                    trade_in_progress_message = "Contract active, awaiting outcome (next tick)..."

                    # For TRADE_DURATION_SECONDS = 1, we only need to wait for the next single tick.
                    # The existing loop will correctly handle this as current_tick_count_for_trade
                    # will go from 0 to 1, then the loop will exit.
                    
                    outcome_tick_price = "N/A"
                    outcome_tick_digit = -1
                    
                    outcome_progress_task_id = global_progress.add_task(f"[yellow]Awaiting outcome tick ({TRADE_DURATION_SECONDS} tick)...", total=TRADE_DURATION_SECONDS)
                    current_tick_count_for_trade = 0

                    while current_tick_count_for_trade < TRADE_DURATION_SECONDS:
                        try:
                            # We are waiting for the *next* tick that comes in from Deriv's subscription.
                            # This will be the tick that determines the outcome for a 1-tick duration.
                            tick_for_outcome = await asyncio.wait_for(message_queue.get(), timeout=5) # Max 5s per tick
                            if tick_for_outcome.get("msg_type") == "tick":
                                current_tick_count_for_trade += 1
                                outcome_tick_price = tick_for_outcome['tick']['quote']
                                outcome_tick_digit = get_last_digit(outcome_tick_price)
                                global_progress.update(outcome_progress_task_id, completed=current_tick_count_for_trade)
                                logging.debug(f"Consumed tick {current_tick_count_for_trade}/{TRADE_DURATION_SECONDS} for outcome.")
                                # Since duration is 1, we break after the first relevant tick.
                                if current_tick_count_for_trade == TRADE_DURATION_SECONDS:
                                    break
                            elif "error" in tick_for_outcome:
                                add_log_message(f"[red]Error while consuming ticks for trade outcome: {tick_for_outcome['error']['message']}[/red]")
                        except asyncio.TimeoutError:
                            add_log_message(f"[bold red]Timeout waiting for outcome tick for contract ID {contract_id}.[/bold red]")
                            outcome_tick_price = "N/A (Timeout)"
                            break
                        except Exception as e:
                            add_log_message(f"[bold red]Error consuming ticks for trade outcome: {e}[/bold red]")
                            outcome_tick_price = "N/A (Error)"
                            break
                    
                    global_progress.remove_task(outcome_progress_task_id)
                    
                    win = False
                    profit_amount = 0.0

                    if outcome_tick_digit != -1: # Only proceed if we got a valid outcome digit
                        if prediction_type == "Over":
                            if outcome_tick_digit > digit_barrier:
                                win = True
                                # Profit calculation based on payout - stake
                                profit_amount = proposal['proposal']['payout'] - STAKE
                            else:
                                win = False
                                profit_amount = -STAKE
                        elif prediction_type == "Under":
                            if outcome_tick_digit < digit_barrier:
                                win = True
                                # Profit calculation based on payout - stake
                                profit_amount = proposal['proposal']['payout'] - STAKE
                            else:
                                win = False
                                profit_amount = -STAKE
                    else: # Timeout or error in getting outcome
                        profit_amount = -STAKE # Assume loss
                        win = False
                        add_log_message(f"[bold red]Failed to determine outcome for contract {contract_id}. Assuming loss.[/bold red]")


                    account_balance += (profit_amount + STAKE)

                    trade_result = "Win" if win else "Loss"
                    trade_result_msg = f"Trade {'[bold green]WIN![/bold green]' if win else '[bold red]LOSS![/bold red]'} P/L: {'[green]' if win else '[red]'}{profit_amount:+.2f}[/]"
                    add_log_message(trade_result_msg)

                    total_profit += profit_amount
                    daily_profit += profit_amount
                    total_trades += 1

                    trade_history.append({
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "prediction": f"{prediction_type} {predicted_digit}",
                        "barrier_digit": digit_barrier,
                        "entry": entry_tick_digit,
                        "outcome": f"{outcome_tick_digit}",
                        "result": trade_result,
                        "profit": profit_amount,
                        "win": win
                    })

                    if daily_profit <= -DAILY_LOSS_LIMIT:
                        trading_active = False
                        add_log_message(f"[bold red]DAILY LOSS LIMIT (${DAILY_LOSS_LIMIT:.2f}) HIT! Current daily profit: ${daily_profit:.2f}. Trading paused.[/bold red]")
                    elif daily_profit >= DAILY_PROFIT_TARGET:
                        trading_active = False
                        add_log_message(f"[bold green]DAILY PROFIT TARGET (${DAILY_PROFIT_TARGET:.2f}) HIT! Current daily profit: ${daily_profit:.2f}. Trading paused.[/bold green]")

                    trade_in_progress_message = "Trade complete. Waiting for next cycle."
                    await asyncio.sleep(0.1) # Short pause to allow UI update and prevent tight loop

                    trade_in_progress_message = "Waiting for next trade..."

        except asyncio.CancelledError:
            current_status_message = "Shutting down gracefully..."
            add_log_message("[bold yellow]🚨 Ctrl+C detected. Initiating graceful shutdown...[/bold yellow]")

            global_progress.stop()
            add_log_message("✅ [yellow]Progress tasks stopped.[/yellow]")

            if receiver_task and not receiver_task.done():
                receiver_task.cancel()
                try:
                    await receiver_task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    add_log_message(f"[red]Error awaiting cancelled receiver task: {e}[/red]")
                add_log_message("✅ [yellow]Receiver task cancelled.[/yellow]")

            if ws and ws.open:
                try:
                    await ws.close()
                    add_log_message("🔌 [yellow]WebSocket connection closed.[/yellow]")
                except Exception as e:
                    add_log_message(f"[red]Error closing Deriv WS: {e}[/red]")

            if model_ws_connection and model_ws_connection.open:
                try:
                    await model_ws_connection.close()
                    add_log_message("🧠 [yellow]Model WS connection closed.[/yellow]")
                except Exception as e:
                    add_log_message(f"[red]Error closing Model WS: {e}[/red]")

            if model_process and model_process.poll() is None:
                try:
                    model_process.terminate()
                    await asyncio.sleep(1)
                    if model_process.poll() is None:
                        model_process.kill()
                    add_log_message("🧠 [yellow]Model engine terminated.[/yellow]")
                except ProcessLookupError:
                    pass
                except Exception as e:
                    add_log_message(f"[red]Error terminating model process: {e}[/red]")

            if os.path.exists(MODEL_READY_FILE):
                try:
                    os.remove(MODEL_READY_FILE)
                    add_log_message("🗑️ [yellow]Model ready flag file removed.[/yellow]")
                except Exception as e:
                    add_log_message(f"[red]Error removing model ready file: {e}[/red]")

        except ConnectionClosedOK:
            current_status_message = "Disconnected."
            add_log_message("[bold red]🔌 WebSocket connection closed gracefully.[/bold red]")
        except Exception as e:
            current_status_message = "BOT CRASHED!"
            add_log_message(f"[bold red]🔥🔥 CRITICAL ERROR: {e}🔥🔥[/bold red]")
        finally:
            main_layout["stats_panel"].update(build_stats_panel())
            main_layout["status_panel"].update(build_status_panel())
            main_layout["info_log_panel"].update(build_info_log_panel())
            main_layout["trade_history_panel"].update(build_trade_history_panel())
            live.refresh()
            console.print("\n[bold red]Bot shutdown complete.[/bold red]")
            await asyncio.sleep(3)

if __name__ == "__main__":
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        console.print("\n[bold red]Bot process interrupted by user (Ctrl+C).[/bold red]")
    except Exception as e:

        console.print(f"\n[bold red]An unexpected error occurred outside the main loop: {str(e)}[/bold red]")
