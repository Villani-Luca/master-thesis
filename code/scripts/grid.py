"""The full training grid, and where it stands: which runs are done, which are half-trained, which are missing.

    python code/scripts/grid.py status                        # table: finished runs per model x universe x horizon
    python code/scripts/grid.py status --universes dji        # only some cells (also --models, --horizons)
    python code/scripts/grid.py run --workers 3               # train every unfinished task, 3 at a time
    python code/scripts/grid.py run --horizons 1 20 --universes dji nasdaq100 sx5e
    python code/scripts/grid.py run --gpus 0 1 --workers 8    # 8 tasks on each of two GPUs
    python code/scripts/grid.py run --gpus 0 --workers 8 --max-ram 128 --cpus 0-15
    python code/scripts/grid.py pending                       # indices of the unfinished tasks (e.g. for a Slurm array)
    python code/scripts/grid.py args 17                       # train_rolling.py arguments of task 17

One task = one (model, universe, horizon, seed), trained on all its test years. `run` is the way to
train on a plain GPU server: start it inside `screen`, and after any interruption (Ctrl-C, logout
without screen, reboot) start the same command again. Finished runs are skipped, and runs stopped
in the middle resume from their last epoch (resume.pt, see xaifin.training.trainer.fit). Each task
logs to logs/grid/task<N>.log. A task starts only while its expected RAM fits in --max-ram; the live
state is in logs/grid/status.json, and `touch logs/grid/STOP` lets the running tasks end and starts no new one.

When two machines share the grid, give them different --universes or --horizons, so they never
train the same task at the same time.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path

import pandas as pd

from xaifin.config import CODE_ROOT, MODEL_GROUPS, RESULTS_ROOT, SEEDS, SL_PL_CONFIGS, TEST_YEARS, UNIVERSES

MODELS = [m for group in MODEL_GROUPS.values() for m in group]
TASKS = list(product(MODELS, UNIVERSES, SL_PL_CONFIGS, SEEDS))  # the order defines the array index


def test_years(universe: str) -> list[int]:
    """No 2020 for the EU universes: their membership history starts in September 2020."""
    return [y for y in TEST_YEARS if not (universe in ("sx5e", "sxxp") and y == 2020)]


def run_state(model, universe, sl_pl, seed, year) -> str:
    run_dir = (RESULTS_ROOT / "Regression" / model / universe / f"sl{sl_pl[0]}_pl{sl_pl[1]}" / f"seed{seed}"
               / f"y{year}")
    if (run_dir / "metrics.json").exists():
        return "done"
    return "partial" if (run_dir / "resume.pt").exists() else "missing"


def task_args(index: int) -> list[str]:
    model, universe, (sl, pl), seed = TASKS[index]
    return ["--models", model, "--universes", universe, "--years", *map(str, test_years(universe)),
            "--seeds", str(seed), "--sl", str(sl), "--pl", str(pl)]


def task_table(universes=None, models=None, horizons=None) -> pd.DataFrame:
    rows = []
    for index, (model, universe, sl_pl, seed) in enumerate(TASKS):
        if ((universes and universe not in universes) or (models and model not in models)
                or (horizons and sl_pl[1] not in horizons)):
            continue
        states = [run_state(model, universe, sl_pl, seed, y) for y in test_years(universe)]
        rows.append({"task": index, "model": model, "universe": universe, "L": sl_pl[1], "seed": seed,
                     "done": states.count("done"), "partial": states.count("partial"), "years": len(states)})
    return pd.DataFrame(rows)


def compress(indices: list[int]) -> str:
    """[1, 2, 3, 7] -> '1-3,7', the --array syntax."""
    parts, i = [], 0
    while i < len(indices):
        j = i
        while j + 1 < len(indices) and indices[j + 1] == indices[j] + 1:
            j += 1
        parts.append(str(indices[i]) if i == j else f"{indices[i]}-{indices[j]}")
        i = j + 1
    return ",".join(parts)


# Rough minutes per run on the RTX 5090 (results/analysis/metrics_summary.csv, L = 5), to start the
# longest tasks first: a long task started last would keep one worker busy while the others idle.
MINUTES_PER_RUN = {"FactorVAE": 67, "DiscoverPLF": 24, "MATCC": 16, "HIST": 11, "FinFormer": 5, "MASTER": 2}
UNIVERSE_WEIGHT = {"dji": 1, "sx5e": 1, "nasdaq100": 1.2, "sp500": 2, "sxxp": 2}


# Peak RSS of one task in GiB, the highest measured over the six models on an H200 server (Oct 2026);
# FactorVAE stays near 30 GiB on the big universes. RAM, not the GPU, limits how many tasks run at once.
PEAK_RAM_GIB = {"dji": 6, "sx5e": 10, "nasdaq100": 17, "sp500": 63, "sxxp": 72}
PEAK_RAM_GIB_MODEL = {("FactorVAE", "sp500"): 30, ("FactorVAE", "sxxp"): 30}
RAM_MARGIN = 1.15     # on top of the measured peak
ADMIT_SHARE = 0.9     # a task starts only if the reserved RAM stays under this share of --max-ram
KILL_SHARE = 0.97     # above this share of --max-ram (real RSS), the youngest task is stopped and requeued
HEAD_WAIT_MIN = 30    # after this long, a big task waiting for RAM stops smaller ones from overtaking it
POLL_S = 10


def ram_budget_gib() -> float | None:
    """80% of the RAM available now, or of the cgroup limit if lower; None where /proc is missing."""
    try:
        meminfo = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    except OSError:
        return None
    available = int(meminfo["MemAvailable"].split()[0]) / 2**20
    try:
        limit = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if limit != "max":
            available = min(available, int(limit) / 2**30)
    except OSError:
        pass
    return 0.8 * available


def tree_rss_gib(pid: int) -> float:
    """Resident memory of a process and its children (Linux; 0 elsewhere)."""
    total, stack = 0, [pid]
    while stack:
        p = stack.pop()
        try:
            with open(f"/proc/{p}/status") as status:
                total += sum(int(line.split()[1]) for line in status if line.startswith("VmRSS:"))
            for thread in os.listdir(f"/proc/{p}/task"):
                with open(f"/proc/{p}/task/{thread}/children") as children:
                    stack += map(int, children.read().split())
        except OSError:
            pass
    return total / 2**20


def parse_cpus(text: str) -> set[int]:
    """'86-101,120' -> {86, ..., 101, 120}, the taskset syntax."""
    cpus = set()
    for part in text.split(","):
        first, _, last = part.partition("-")
        cpus.update(range(int(first), int(last or first) + 1))
    return cpus


def run_tasks(tasks: list[int], workers: int, threads: int, device: str | None, gpus: list[str] | None,
              max_ram: float | None = None, cpus: set[int] | None = None) -> None:
    """Train the tasks, each in its own train_rolling.py process, `workers` at a time on each GPU.

    A task starts when a GPU slot is free and its expected peak RAM (PEAK_RAM_GIB, raised to what it
    is seen using) fits in `max_ram` GiB together with the running ones. If the real RAM still grows
    too close to `max_ram`, the youngest task is stopped and requeued: it resumes from its last epoch
    with identical results. `cpus` pins every process to those CPU ids. While running, the state is in
    logs/grid/status.json; creating logs/grid/STOP lets the running tasks finish and starts no new one.
    Failed tasks are not retried (a Python error is to be reported, not hidden).
    """
    log_dir = CODE_ROOT.parent / "logs" / "grid"
    log_dir.mkdir(parents=True, exist_ok=True)
    status_file, stop_file = log_dir / "status.json", log_dir / "STOP"
    stop_file.unlink(missing_ok=True)  # left by the previous run
    script = str(CODE_ROOT / "scripts" / "train_rolling.py")
    waiting = sorted(tasks, key=lambda i: -MINUTES_PER_RUN[TASKS[i][0]] * UNIVERSE_WEIGHT[TASKS[i][1]])
    slots = [gpu for _ in range(workers) for gpu in gpus or [None]]
    expected = {i: RAM_MARGIN * PEAK_RAM_GIB_MODEL.get(TASKS[i][:2], PEAK_RAM_GIB[TASKS[i][1]]) for i in tasks}
    running = {}  # task -> (process, gpu, start time, log file)
    stopped, ok, failed = set(), [], []
    head_waiting_since = None
    env = os.environ | {"OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)}
    pin = (lambda: os.sched_setaffinity(0, cpus)) if cpus else None

    def say(text: str) -> None:
        print(f"{datetime.now():%m-%d %H:%M:%S}  {text}", flush=True)

    def start(index: int) -> None:
        gpu = slots.pop()
        task_env = env | ({"CUDA_VISIBLE_DEVICES": gpu} if gpu is not None else {})
        command = [sys.executable, script, *task_args(index)] + (["--device", device] if device else [])
        log = open(log_dir / f"task{index}.log", "a", encoding="utf-8")
        process = subprocess.Popen(command, env=task_env, stdout=log, stderr=subprocess.STDOUT, preexec_fn=pin)
        running[index] = (process, gpu, datetime.now(), log)
        where = f" on GPU {gpu}" if gpu is not None else ""
        say(f"start task {index}{where} (~{expected[index]:.0f} GiB): {' '.join(task_args(index))}")

    budget = f", RAM budget {max_ram:.0f} GiB" if max_ram else ""
    say(f"{len(waiting)} tasks to train, at most {len(slots)} at a time{budget}; logs in {log_dir}")
    try:
        while waiting or running:
            for index, (process, gpu, _, log) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                slots.append(gpu)
                del running[index]
                if index in stopped:
                    stopped.discard(index)
                    waiting.insert(0, index)
                    say(f"requeued task {index} (stopped for RAM, resumes from its last epoch)")
                elif code == 0:
                    ok.append(index)
                    say(f"done task {index}")
                else:
                    failed.append(index)
                    say(f"FAILED (exit {code}) task {index} -> {log_dir / f'task{index}.log'}")

            rss = {i: tree_rss_gib(process.pid) for i, (process, *_) in running.items()}
            for i, used in rss.items():
                expected[i] = max(expected[i], RAM_MARGIN * used)
            active = [i for i in running if i not in stopped]
            if max_ram and sum(rss.values()) > KILL_SHARE * max_ram and len(active) > 1:
                youngest = max(active, key=lambda i: running[i][2])
                running[youngest][0].terminate()
                stopped.add(youngest)
                say(f"RAM {sum(rss.values()):.0f} GiB > {KILL_SHARE:.0%} of {max_ram:.0f}: stopping task {youngest}")

            stop = stop_file.exists()
            if stop and not running:
                say(f"{stop_file} found: no new tasks started")
                break
            for position, index in enumerate(list(waiting) if not stop else []):
                if not slots:
                    break
                reserved = sum(expected[i] for i in running)
                if not running or not max_ram or reserved + expected[index] <= ADMIT_SHARE * max_ram:
                    waiting.remove(index)
                    start(index)
                    if position == 0:
                        head_waiting_since = None
                elif position == 0:
                    head_waiting_since = head_waiting_since or datetime.now()
                    if (datetime.now() - head_waiting_since).total_seconds() > 60 * HEAD_WAIT_MIN:
                        break  # no more overtaking: let the RAM free up for it

            status_file.write_text(json.dumps({
                "time": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "stop_file": stop, "ram_budget_gib": max_ram,
                "ram_used_gib": round(sum(rss.values()), 1), "waiting": len(waiting), "ok": ok, "failed": failed,
                "running": [{"task": i, "args": " ".join(task_args(i)), "gpu": gpu,
                             "minutes": round((datetime.now() - t).total_seconds() / 60),
                             "rss_gib": round(rss.get(i, 0), 1)} for i, (_, gpu, t, _) in running.items()],
            }, indent=1))
            time.sleep(POLL_S)
    except KeyboardInterrupt:
        for process, *_ in running.values():
            process.terminate()
        for process, *_ in running.values():
            process.wait()
        raise
    say(f"finished: {len(ok)} ok, {len(failed)} failed{f' {failed}' if failed else ''}, {len(waiting)} not started"
        f" (run again to retry or continue)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["status", "run", "pending", "args"])
    parser.add_argument("task", nargs="?", type=int, help="task index (for 'args')")
    parser.add_argument("--universes", nargs="+", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--horizons", nargs="+", type=int, default=None, help="horizons L, e.g. 1 20")
    parser.add_argument("--workers", type=int, default=2, help="run: tasks trained at the same time (per GPU with --gpus)")
    parser.add_argument("--gpus", nargs="+", default=None, help="run: GPU ids to spread the tasks over, e.g. 0 1")
    parser.add_argument("--threads", type=int, default=2,
                        help="run: CPU threads per task (2 and 6 give bit-identical results at the same speed)")
    parser.add_argument("--max-ram", type=float, default=None,
                        help="run: GiB of RAM the tasks may use together (default: 80%% of the available RAM)")
    parser.add_argument("--cpus", default=None, help="run: pin every task to these CPU ids, e.g. 0-15")
    parser.add_argument("--device", default=None, help="run: cuda or cpu (default: cuda if available)")
    args = parser.parse_args()

    if args.command == "args":
        print(" ".join(task_args(args.task)))
        return

    table = task_table(args.universes, args.models, args.horizons)
    unfinished = table.loc[table["done"] < table["years"], "task"].tolist()
    if args.command == "pending":
        print(compress(unfinished))
        return
    if args.command == "run":
        run_tasks(unfinished, args.workers, args.threads, args.device, args.gpus,
                  args.max_ram or ram_budget_gib(), parse_cpus(args.cpus) if args.cpus else None)
        return

    runs = table.groupby(["universe", "L", "model"], sort=False)[["done", "partial", "years"]].sum()
    runs["done"] = runs["done"].astype(str) + "/" + (runs["years"] * 1).astype(str)
    grid = runs["done"].unstack("model")[[m for m in MODELS if m in runs.index.get_level_values("model")]]
    print("Finished runs (test years x seeds) per cell:\n")
    print(grid.to_string())
    done, total, partial = table["done"].sum(), table["years"].sum(), table["partial"].sum()
    print(f"\n{done}/{total} runs finished ({done / total:.0%}), {partial} half-trained (will resume), "
          f"{(table['done'] < table['years']).sum()} of {len(table)} tasks still to run")


if __name__ == "__main__":
    main()
